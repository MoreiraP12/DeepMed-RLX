// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { useCallback, useRef, useState, useEffect } from "react";

import { LoadingAnimation } from "~/components/deer-flow/loading-animation";
import { Markdown } from "~/components/deer-flow/markdown";
import { MarkdownWithThinking } from "~/components/deer-flow/markdown-with-thinking";
import ReportEditor from "~/components/editor";
import { useReplay } from "~/core/replay";
import { useMessage, useStore } from "~/core/store";
import { cn } from "~/lib/utils";
import { EBMPyramid } from "~/components/deer-flow/ebm-pyramid";
import { useTheme } from "next-themes";
import { TavilySourcesSection } from "./tavily-sources-section";

export function ResearchReportBlock({
  className,
  messageId,
  editing,
}: {
  className?: string;
  researchId: string;
  messageId: string;
  editing: boolean;
}) {
  const { theme } = useTheme();
  const [isReplay, setIsReplay] = useState(false);
  const { openReplayModal } = useReplay();
  const contentRef = useRef<HTMLDivElement>(null);

  const message = useMessage(messageId);

  const isCompleted = !message?.isStreaming && (message?.finishReason === "stop" || message?.finishReason === "interrupt");

  const handleMarkdownChange = useCallback(
    (markdown: string) => {
      if (message) {
        useStore.getState().updateMessageContent(messageId, markdown);
      }
    },
    [message, messageId],
  );

  useEffect(() => {
    const isReplayPath = window.location.pathname.includes("/replay");
    setIsReplay(isReplayPath);
  }, []);

  // Enhanced component for rendering EBM pyramid with React component
  const EBMPyramidRenderer = ({ className, children, ...props }: any) => {
    // Check if this is the special ebm-pyramid-data code block
    if (props['data-language'] === 'ebm-pyramid-data' && children) {
      try {
        // Parse the JSON data from the code block
        const jsonData = JSON.parse(children);
        
        return (
          <div className="my-8">
            {/* Enhanced EBM Pyramid Header */}
            <div className="text-center mb-6">
              <div className="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium bg-gradient-to-r from-emerald-100 to-blue-100 text-emerald-800 dark:from-emerald-900/20 dark:to-blue-900/20 dark:text-emerald-200">
                📊 Interactive Evidence Quality Analysis
              </div>
              <p className="mt-2 text-sm text-muted-foreground max-w-2xl mx-auto">
                Enhanced Evidence-Based Medicine pyramid with Tavily source analysis, logos, and quality metrics
              </p>
            </div>
            
            {/* React EBM Pyramid Component with JSON data */}
            <EBMPyramid data={jsonData} />
          </div>
        );
      } catch (error) {
        console.error('Error parsing EBM pyramid data:', error);
        // Fallback to regular code block if JSON parsing fails
        return <code className={className} {...props}>{children}</code>;
      }
    }
    
    // Default code block rendering
    return <code className={className} {...props}>{children}</code>;
  };

  // Enhanced component for rendering images (fallback for any remaining image-based EBM pyramids)
  const EBMImageRenderer = ({ src, alt }: { src: string; alt: string }) => {
    // Check if this is an EBM pyramid image (fallback case)
    if (src.includes('ebm_pyramid') || alt.toLowerCase().includes('ebm pyramid')) {
      return (
        <div className="my-8 p-4 border rounded-lg bg-accent/30">
          <details className="cursor-pointer">
            <summary className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
              View Static Pyramid Image (Fallback)
            </summary>
            <div className="mt-3">
              <img 
                src={src} 
                alt={alt}
                className="mx-auto max-w-full h-auto rounded-md shadow-lg"
                style={{ maxHeight: '600px' }}
              />
            </div>
          </details>
        </div>
      );
    }
    
    // Default image rendering for non-EBM images
    return <img src={src} alt={alt} className="mx-auto max-w-full h-auto rounded-md" />;
  };

  // TODO: scroll to top when completed, but it's not working
  // useEffect(() => {
  //   if (isCompleted && contentRef.current) {
  //     setTimeout(() => {
  //       contentRef
  //         .current!.closest("[data-radix-scroll-area-viewport]")
  //         ?.scrollTo({
  //           top: 0,
  //           behavior: "smooth",
  //         });
  //     }, 500);
  //   }
  // }, [isCompleted]);

  return (
    <div
      ref={contentRef}
      className={cn("relative flex flex-col pt-4 pb-8", className)}
    >
      {!isReplay && isCompleted && editing ? (
        <ReportEditor
          content={message?.content}
          onMarkdownChange={handleMarkdownChange}
        />
      ) : (
        <>
          <MarkdownWithThinking 
            animated 
            checkLinkCredibility
            components={{
              code: EBMPyramidRenderer, // Use custom EBM pyramid code block component
              img: EBMImageRenderer, // Use custom EBM pyramid image component (fallback)
            }}
          >
            {message?.content}
          </MarkdownWithThinking>
          {message?.isStreaming && <LoadingAnimation className="my-12" />}
          
          {/* Add Tavily Sources Section at the end (only when not streaming) */}
          {!message?.isStreaming && (
            <TavilySourcesSection messageId={messageId} />
          )}
        </>
      )}
    </div>
  );
}
