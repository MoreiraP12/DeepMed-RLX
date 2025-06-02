// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { ChevronDown, ChevronRight } from "lucide-react";
import { useMemo, useState } from "react";
import { cn } from "~/lib/utils";
import { Markdown } from "./markdown";

interface MarkdownWithThinkingProps {
  children?: string;
  className?: string;
  animated?: boolean;
  checkLinkCredibility?: boolean;
  enableCopy?: boolean;
  style?: React.CSSProperties;
}

export function MarkdownWithThinking({
  children,
  className,
  animated = false,
  checkLinkCredibility = false,
  enableCopy,
  style,
}: MarkdownWithThinkingProps) {
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(false);

  const { mainContent, thinkingContent } = useMemo(() => {
    if (!children) return { mainContent: "", thinkingContent: "" };

    // Extract content between <think> and </think> tags
    const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
    const matches = Array.from(children.matchAll(thinkRegex));
    
    if (matches.length === 0) {
      return { mainContent: children, thinkingContent: "" };
    }

    // Extract thinking content and remove think tags from main content
    const thinkingContent = matches.map(match => match[1]).join('\n\n');
    const mainContent = children.replace(thinkRegex, '').trim();

    return { mainContent, thinkingContent };
  }, [children]);

  const hasThinkingContent = thinkingContent.trim().length > 0;

  return (
    <div className={className}>
      {hasThinkingContent && (
        <div className="mb-4">
          <button
            onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors group"
          >
            {isThinkingExpanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
            <span className="group-hover:underline">
              {isThinkingExpanded ? "Hide" : "Show"} thinking process
            </span>
          </button>
          
          {isThinkingExpanded && (
            <div className="mt-3 p-4 bg-muted/30 rounded-lg border border-muted">
              <div className="text-muted-foreground text-sm">
                <Markdown
                  className={cn("prose-sm", "opacity-75")}
                  animated={animated}
                  checkLinkCredibility={checkLinkCredibility}
                >
                  {thinkingContent}
                </Markdown>
              </div>
            </div>
          )}
        </div>
      )}
      
      <Markdown
        className="prose dark:prose-invert"
        style={style}
        animated={animated}
        checkLinkCredibility={checkLinkCredibility}
        enableCopy={enableCopy}
      >
        {mainContent}
      </Markdown>
    </div>
  );
} 