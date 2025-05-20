import { useMemo } from "react";
import { useStore, useToolCalls } from "~/core/store";
import { Tooltip } from "./tooltip";
import { WarningFilled } from "@ant-design/icons";
import { parseJSON } from "~/core/utils";

export const Link = ({
  href,
  children,
  checkLinkCredibility = false,
}: {
  href: string | undefined;
  children: React.ReactNode;
  checkLinkCredibility: boolean;
}) => {
  const toolCalls = useToolCalls();
  const responding = useStore((state) => state.responding);

  const credibleLinks = useMemo(() => {
    const links = new Set<string>();
    if (!checkLinkCredibility) return links;

    (toolCalls || []).forEach((call) => {
      if (call && call.name === "web_search" && call.result) {
        try {
          // Log the raw result
          console.warn("[Link] web_search call.result (raw):", call.result);
          
          // Try to parse it with the robust parseJSON utility
          const result = parseJSON(call.result, []);
          console.warn("[Link] web_search call.result (parsed):", result);
          
          // If the result is an array and contains objects with url property
          if (Array.isArray(result)) {
            result.forEach((r) => {
              if (r && typeof r === 'object' && 'url' in r) {
                links.add(r.url);
              }
            });
          } 
          // If the parsing returned an empty array (fallback), try to extract URLs directly
          else if (typeof call.result === 'string') {
            // Look for URLs in the text content
            const urlRegex = /(https?:\/\/[^\s]+)/g;
            const matches = call.result.match(urlRegex);
            if (matches) {
              matches.forEach(url => links.add(url));
              console.warn("[Link] Extracted URLs from text:", matches);
            }
          }
        } catch (err) {
          console.error("[Link] Failed to process web_search call.result:", err);
        }
      }
    });
    return links;
  }, [toolCalls]);

  const isCredible = useMemo(() => {
    return checkLinkCredibility && href && !responding
      ? credibleLinks.has(href)
      : true;
  }, [credibleLinks, href, responding, checkLinkCredibility]);

  return (
    <span className="flex items-center gap-1.5">
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
      {!isCredible && (
        <Tooltip
          title="This link might be a hallucination from AI model and may not be reliable."
          delayDuration={300}
        >
          <WarningFilled className="text-sx transition-colors hover:!text-yellow-500" />
        </Tooltip>
      )}
    </span>
  );
};
