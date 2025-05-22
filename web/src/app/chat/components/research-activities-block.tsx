// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { PythonOutlined } from "@ant-design/icons";
import { motion } from "framer-motion";
import { LRUCache } from "lru-cache";
import { BookOpenText, PencilRuler, Search } from "lucide-react";
import { useTheme } from "next-themes";
import { useMemo } from "react";
import SyntaxHighlighter from "react-syntax-highlighter";
import { docco } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { dark } from "react-syntax-highlighter/dist/esm/styles/prism";

import { FavIcon } from "~/components/deer-flow/fav-icon";
import Image from "~/components/deer-flow/image";
import { LoadingAnimation } from "~/components/deer-flow/loading-animation";
import { Markdown } from "~/components/deer-flow/markdown";
import { RainbowText } from "~/components/deer-flow/rainbow-text";
import { Tooltip } from "~/components/deer-flow/tooltip";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "~/components/ui/accordion";
import { Skeleton } from "~/components/ui/skeleton";
import { findMCPTool } from "~/core/mcp";
import type { ToolCallRuntime } from "~/core/messages";
import { useMessage, useStore } from "~/core/store";
import { parseJSON } from "~/core/utils";
import { cn } from "~/lib/utils";

export function ResearchActivitiesBlock({
  className,
  researchId,
}: {
  className?: string;
  researchId: string;
}) {
  const activityIds = useStore((state) =>
    state.researchActivityIds.get(researchId),
  )!;
  const ongoing = useStore((state) => state.ongoingResearchId === researchId);
  return (
    <>
      <ul className={cn("flex flex-col py-4", className)}>
        {activityIds.map(
          (activityId, i) =>
            i !== 0 && (
              <motion.li
                key={activityId}
                style={{ transition: "all 0.4s ease-out" }}
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  duration: 0.4,
                  ease: "easeOut",
                }}
              >
                <ActivityMessage messageId={activityId} />
                <ActivityListItem messageId={activityId} />
                {i !== activityIds.length - 1 && <hr className="my-8" />}
              </motion.li>
            ),
        )}
      </ul>
      {ongoing && <LoadingAnimation className="mx-4 my-12" />}
    </>
  );
}

function ActivityMessage({ messageId }: { messageId: string }) {
  const message = useMessage(messageId);
  if (message?.agent && message.content) {
    if (message.agent !== "reporter" && message.agent !== "planner") {
      return (
        <div className="px-4 py-2">
          <Markdown animated checkLinkCredibility>
            {message.content}
          </Markdown>
        </div>
      );
    }
  }
  return null;
}

function ActivityListItem({ messageId }: { messageId: string }) {
  const message = useMessage(messageId);
  if (message) {
    if (!message.isStreaming && message.toolCalls?.length) {
      for (const toolCall of message.toolCalls) {
        if (toolCall.name === "web_search") {
          return <WebSearchToolCall key={toolCall.id} toolCall={toolCall} />;
        } else if (toolCall.name === "crawl_tool") {
          return <CrawlToolCall key={toolCall.id} toolCall={toolCall} />;
        } else if (toolCall.name === "python_repl_tool") {
          return <PythonToolCall key={toolCall.id} toolCall={toolCall} />;
        } else {
          return <MCPToolCall key={toolCall.id} toolCall={toolCall} />;
        }
      }
    }
  }
  return null;
}

const __pageCache = new LRUCache<string, string>({ max: 100 });
type SearchResult =
  | {
      type: "page";
      title: string;
      url: string;
      content: string;
    }
  | {
      type: "image";
      image_url: string;
      image_description: string;
    };
function WebSearchToolCall({ toolCall }: { toolCall: ToolCallRuntime }) {
  const searching = useMemo(() => {
    return toolCall.result === undefined;
  }, [toolCall.result]);
  const searchResults = useMemo<SearchResult[]>(() => {
    let results: SearchResult[] | undefined = undefined;
    try {
      // Log raw result for debugging
      if (toolCall.result) {
        console.warn("[WebSearchToolCall] Raw result:", toolCall.result);
      }

      if (!toolCall.result) {
        return [];
      }
      
      // Try to parse using parseJSON
      results = parseJSON(toolCall.result, []);
      console.warn("[WebSearchToolCall] Parsed result:", results);
      
      // Handle non-standard search results
      if (!Array.isArray(results) || results.length === 0) {
        // For academic paper results, try to extract information
        if (toolCall.result.includes('Published:') && toolCall.result.includes('Title:')) {
          const papers = toolCall.result.split(/\n\n(?=Published:)/);
          results = papers.map(paper => {
            const titleMatch = paper.match(/Title: (.*)/);
            const title = titleMatch ? titleMatch[1] : "Academic paper";
            return {
              type: "page",
              title: title,
              url: `https://scholar.google.com/scholar?q=${encodeURIComponent(title)}`,
              content: paper
            };
          });
          console.warn("[WebSearchToolCall] Created academic paper results:", results);
        }
      }
    } catch (err) {
      console.error("[WebSearchToolCall] Error parsing result:", err, toolCall.result);
      results = [];
    }
    
    // Ensure we have a valid array
    if (!Array.isArray(results)) {
      results = [];
    }
    
    // Cache page titles if available
    results.forEach((result) => {
      if (result.type === "page") {
        __pageCache.set(result.url, result.title);
      }
    });
    
    return results;
  }, [toolCall.result]);
  const pageResults = useMemo(
    () => searchResults?.filter((result) => result.type === "page"),
    [searchResults],
  );
  const imageResults = useMemo(
    () => searchResults?.filter((result) => result.type === "image"),
    [searchResults],
  );
  return (
    <section className="mt-4 pl-4">
      <div className="font-medium italic">
        <RainbowText
          className="flex items-center"
          animated={searchResults === undefined}
        >
          <Search size={16} className={"mr-2"} />
          <span>Searching for&nbsp;</span>
          <span className="max-w-[500px] overflow-hidden text-ellipsis whitespace-nowrap">
            {(toolCall.args as { query: string }).query}
          </span>
        </RainbowText>
      </div>
      <div className="pr-4">
        {pageResults && (
          <ul className="mt-2 flex flex-wrap gap-4">
            {searching &&
              [...Array(6)].map((_, i) => (
                <li
                  key={`search-result-${i}`}
                  className="flex h-40 w-40 gap-2 rounded-md text-sm"
                >
                  <Skeleton
                    className="to-accent h-full w-full rounded-md bg-gradient-to-tl from-slate-400"
                    style={{ animationDelay: `${i * 0.2}s` }}
                  />
                </li>
              ))}
            {pageResults
              .filter((result) => result.type === "page")
              .map((searchResult, i) => (
                <motion.li
                  key={`search-result-${i}`}
                  className="text-muted-foreground bg-accent flex max-w-40 gap-2 rounded-md px-2 py-1 text-sm"
                  initial={{ opacity: 0, y: 10, scale: 0.66 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{
                    duration: 0.2,
                    delay: i * 0.1,
                    ease: "easeOut",
                  }}
                >
                  <FavIcon
                    className="mt-1"
                    url={searchResult.url}
                    title={searchResult.title}
                  />
                  <a href={searchResult.url} target="_blank">
                    {searchResult.title}
                  </a>
                </motion.li>
              ))}
            {imageResults.map((searchResult, i) => (
              <motion.li
                key={`search-result-${i}`}
                initial={{ opacity: 0, y: 10, scale: 0.66 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{
                  duration: 0.2,
                  delay: i * 0.1,
                  ease: "easeOut",
                }}
              >
                <a
                  className="flex flex-col gap-2 overflow-hidden rounded-md opacity-75 transition-opacity duration-300 hover:opacity-100"
                  href={searchResult.image_url}
                  target="_blank"
                >
                  <Image
                    src={searchResult.image_url}
                    alt={searchResult.image_description}
                    className="bg-accent h-40 w-40 max-w-full rounded-md bg-cover bg-center bg-no-repeat"
                    imageClassName="hover:scale-110"
                    imageTransition
                  />
                </a>
              </motion.li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

function CrawlToolCall({ toolCall }: { toolCall: ToolCallRuntime }) {
  const url = useMemo(
    () => (toolCall.args as { url: string }).url,
    [toolCall.args],
  );
  const title = useMemo(() => __pageCache.get(url), [url]);
  return (
    <section className="mt-4 pl-4">
      <div>
        <RainbowText
          className="flex items-center text-base font-medium italic"
          animated={toolCall.result === undefined}
        >
          <BookOpenText size={16} className={"mr-2"} />
          <span>Reading</span>
        </RainbowText>
      </div>
      <ul className="mt-2 flex flex-wrap gap-4">
        <motion.li
          className="text-muted-foreground bg-accent flex h-40 w-40 gap-2 rounded-md px-2 py-1 text-sm"
          initial={{ opacity: 0, y: 10, scale: 0.66 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{
            duration: 0.2,
            ease: "easeOut",
          }}
        >
          <FavIcon className="mt-1" url={url} title={title} />
          <a
            className="h-full flex-grow overflow-hidden text-ellipsis whitespace-nowrap"
            href={url}
            target="_blank"
          >
            {title ?? url}
          </a>
        </motion.li>
      </ul>
    </section>
  );
}

function PythonToolCall({ toolCall }: { toolCall: ToolCallRuntime }) {
  const code = useMemo<string>(() => {
    return (toolCall.args as { code: string }).code;
  }, [toolCall.args]);
  const { resolvedTheme } = useTheme();
  return (
    <section className="mt-4 pl-4">
      <div className="flex items-center">
        <PythonOutlined className={"mr-2"} />
        <RainbowText
          className="text-base font-medium italic"
          animated={toolCall.result === undefined}
        >
          Running Python code
        </RainbowText>
      </div>
      <div>
        <div className="bg-accent mt-2 max-h-[400px] max-w-[calc(100%-120px)] overflow-y-auto rounded-md p-2 text-sm">
          <SyntaxHighlighter
            language="python"
            style={resolvedTheme === "dark" ? dark : docco}
            customStyle={{
              background: "transparent",
              border: "none",
              boxShadow: "none",
            }}
          >
            {code.trim()}
          </SyntaxHighlighter>
        </div>
      </div>
      {toolCall.result && <PythonToolCallResult toolCall={toolCall} />}
    </section>
  );
}

function PythonToolCallResult({ toolCall }: { toolCall: ToolCallRuntime }) {
  const { resolvedTheme } = useTheme();
  const result = toolCall.result || "";
  
  // Add console logging for debugging
  console.log("PythonToolCallResult raw result:", result);
  
  const hasError = useMemo(
    
    () =>  JSON.parse(result).result.includes("\nError"),
    [result],
  );
  
  const error = useMemo(() => {
    // Only proceed if hasError is true and the result string exists.
    if (hasError && result) {
      try {
        // Parse the outer JSON string.
        const parsedData = JSON.parse(result);
        const stringToSearch = parsedData.result; // This is the string containing the actual error message.
  
        if (typeof stringToSearch !== 'string') {
          return null; // The expected 'result' property within the JSON was not a string.
        }
  
        const errorMarker = "Error: ";
        // Use lastIndexOf to find the last occurrence of "Error: ",
        // which is more likely to be the specific Python exception message.
        const startIndex = stringToSearch.lastIndexOf(errorMarker);
  
        if (startIndex !== -1) {
          const actualStartIndex = startIndex + errorMarker.length;
          // After JSON.parse, "\\n" from the JSON string becomes an actual newline "\n".
          const endIndex = stringToSearch.indexOf("\n", actualStartIndex);
  
          if (endIndex !== -1) {
            // Extract the first line of the specific error message.
            return stringToSearch.slice(actualStartIndex, endIndex).trim();
          } else {
            // If no newline, the error message might go to the end of the string.
            return stringToSearch.slice(actualStartIndex).trim();
          }
        } else {
          // If "Error: " marker is not found, but hasError is true,
          // the entire stringToSearch is likely the error message.
          return stringToSearch.trim();
        }
      } catch (e) {
        // Handles JSON.parse errors or other processing issues.
        // If hasError is true but parsing failed, you might return the raw 'result'
        // or null depending on how you want to handle malformed error strings.
        // Returning null is safer if the structure is not as expected.
        return null;
      }
    }
    // If not hasError or no result string.
    return null;
  }, [result, hasError]);
  
  const stdout = useMemo(() => {
    if (hasError || !result) {
      return null;
    }
  
    try {
      const parsedData = JSON.parse(result);
      const stringToSearch = parsedData.result;
  
      if (typeof stringToSearch !== 'string') {
        return null;
      }
  
      const stdoutMarker = "Stdout: ";
      const startIndex = stringToSearch.indexOf(stdoutMarker);
  
      if (startIndex !== -1) {
        const actualStartIndex = startIndex + stdoutMarker.length;
        // After JSON.parse, literal "\\n" becomes an actual newline "\n".
        const endIndex = stringToSearch.indexOf("\n", actualStartIndex);
  
        if (endIndex !== -1) {
          return stringToSearch.slice(actualStartIndex, endIndex);
        } else {
          // Assume Stdout content goes to the end if no trailing newline.
          return stringToSearch.slice(actualStartIndex);
        }
      } else {
        // "Stdout: " marker not found in the content string.
        return null;
      }
    } catch (e) {
      // Handles JSON.parse errors or other processing issues.
      return null;
    }
  }, [result, hasError]);
  
  // Log the parsed stdout for debugging
  console.log("Parsed stdout:", stdout);
  
  // Extract image paths from the stdout
  const images = useMemo(() => {
    const imagePaths: string[] = [];
    if (stdout) {
      // Find all image markers in the format [Image saved to: path]
      const imageRegex = /\[Image saved to: (.*?)\]/g;
      let match;
      while ((match = imageRegex.exec(stdout)) !== null) {
        if (match[1]) {
          imagePaths.push(match[1]);
        }
      }
      
      // Try alternate format: directly look for IMAGE: markers
      if (imagePaths.length === 0 && stdout.includes("IMAGE:")) {
        const directImageRegex = /IMAGE:\s+(\S+)/g; // \S+ matches non-whitespace characters
        while ((match = directImageRegex.exec(stdout)) !== null) {
          if (match[1]) {
            imagePaths.push(match[1]);
          }
        }
      }
    }
    
    // Deduplicate the imagePaths array
    const uniqueImagePaths = [...new Set(imagePaths)];
    
    // Log the final unique images
    console.log("Found unique images:", uniqueImagePaths); // You can uncomment this if needed for debugging
    
    return uniqueImagePaths;
  }, [stdout]);
  
  return (
    <>
      <div className="mt-4 font-medium italic">
        {hasError ? "Error when executing the above code" : "Execution output"}
      </div>
      <div className="bg-accent mt-2 max-h-[400px] max-w-[calc(100%-120px)] overflow-y-auto rounded-md p-2 text-sm">
        <SyntaxHighlighter
          language="plaintext"
          style={resolvedTheme === "dark" ? dark : docco}
          customStyle={{
            color: hasError ? "red" : "inherit",
            background: "transparent",
            border: "none",
            boxShadow: "none",
          }}
        >
          {error ?? stdout ?? "(empty)"}
        </SyntaxHighlighter>
      </div>
      
      {/* Display any images found in the output */}
      {images.length > 0 && (
        <div className="mt-4">
          <div className="font-medium italic">Generated Images</div>
          <div className="mt-2 flex flex-wrap gap-4">
            {images.map((imagePath, index) => (
              <motion.div
                key={`python-image-${index}`}
                initial={{ opacity: 0, y: 10, scale: 0.8 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
              >
                <a 
                  href={`./images/${imagePath.replace(/^images\//, '')}`} 
                  target="_blank"
                  className="block rounded-md overflow-hidden shadow-md hover:shadow-lg transition-shadow"
                >
                  <Image
                    src={`/images/${imagePath.replace(/^images\//, '')}`}
                    alt={`Python generated image ${index + 1}`}
                    className="h-48 w-auto max-w-[400px] bg-cover bg-center"
                    imageClassName="hover:scale-105 transition-transform duration-200"
                    imageTransition
                    fallback={<div className="p-4 text-center">Failed to load image</div>}
                  />
                </a>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

function MCPToolCall({ toolCall }: { toolCall: ToolCallRuntime }) {
  const tool = useMemo(() => findMCPTool(toolCall.name), [toolCall.name]);
  const { resolvedTheme } = useTheme();
  return (
    <section className="mt-4 pl-4">
      <div className="w-fit overflow-y-auto rounded-md py-0">
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="item-1">
            <AccordionTrigger>
              <Tooltip title={tool?.description}>
                <div className="flex items-center font-medium italic">
                  <PencilRuler size={16} className={"mr-2"} />
                  <RainbowText
                    className="pr-0.5 text-base font-medium italic"
                    animated={toolCall.result === undefined}
                  >
                    Running {toolCall.name ? toolCall.name + "()" : "MCP tool"}
                  </RainbowText>
                </div>
              </Tooltip>
            </AccordionTrigger>
            <AccordionContent>
              {toolCall.result && (
                <div className="bg-accent max-h-[400px] max-w-[560px] overflow-y-auto rounded-md text-sm">
                  <SyntaxHighlighter
                    language="json"
                    style={resolvedTheme === "dark" ? dark : docco}
                    customStyle={{
                      background: "transparent",
                      border: "none",
                      boxShadow: "none",
                    }}
                  >
                    {toolCall.result.trim()}
                  </SyntaxHighlighter>
                </div>
              )}
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </section>
  );
}
