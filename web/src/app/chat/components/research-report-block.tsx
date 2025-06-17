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

  const isCompleted = message?.isCompleted ?? false;

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
  const EBMPyramidRenderer = ({ src, alt }: { src: string; alt: string }) => {
    // Check if this is an EBM pyramid image
    if (src.includes('ebm_pyramid') || alt.toLowerCase().includes('ebm pyramid')) {
      // Try to find the corresponding JSON data file
      const jsonPath = src.replace(/\.(png|jpg|jpeg)$/i, '_data.json').replace('enhanced_ebm_pyramid', 'ebm_pyramid_data');
      
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
          
          {/* React EBM Pyramid Component */}
          <EBMPyramid jsonPath={jsonPath} />
          
          {/* Fallback to original image if JSON data fails to load */}
          <div className="mt-6 p-4 border rounded-lg bg-accent/30">
            <details className="cursor-pointer">
              <summary className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                View Static Pyramid Image
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
              img: EBMPyramidRenderer, // Use custom EBM pyramid image component
            }}
          >
            {message?.content}
          </MarkdownWithThinking>
          {message?.isStreaming && <LoadingAnimation className="my-12" />}
        </>
      )}
    </div>
  );
}
