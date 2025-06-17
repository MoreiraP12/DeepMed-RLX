// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { useCallback, useRef } from "react";

import { LoadingAnimation } from "~/components/deer-flow/loading-animation";
import { Markdown } from "~/components/deer-flow/markdown";
import { MarkdownWithThinking } from "~/components/deer-flow/markdown-with-thinking";
import ReportEditor from "~/components/editor";
import { useReplay } from "~/core/replay";
import { useMessage, useStore } from "~/core/store";
import { cn } from "~/lib/utils";

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
  const message = useMessage(messageId);
  const { isReplay } = useReplay();
  const handleMarkdownChange = useCallback(
    (markdown: string) => {
      if (message) {
        message.content = markdown;
        useStore.setState({
          messages: new Map(useStore.getState().messages).set(
            message.id,
            message,
          ),
        });
      }
    },
    [message],
  );
  const contentRef = useRef<HTMLDivElement>(null);
  const isCompleted = message?.isStreaming === false && message?.content !== "";
  
  // Custom component for rendering EBM pyramid images with enhanced styling
  const EBMPyramidImage = ({ src, alt }: { src: string; alt: string }) => {
    if (src.includes('ebm_pyramid') || alt.toLowerCase().includes('ebm pyramid')) {
      return (
        <div className="my-6 p-4 border-2 border-blue-200 rounded-lg bg-blue-50 dark:bg-blue-900/20 dark:border-blue-800">
          <div className="text-center mb-3">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800 dark:bg-blue-800 dark:text-blue-200">
              📊 Evidence Quality Analysis
            </span>
          </div>
          <img 
            src={src} 
            alt={alt}
            className="mx-auto max-w-full h-auto rounded-md shadow-lg"
            style={{ maxHeight: '600px' }}
          />
          <div className="mt-3 text-sm text-gray-600 dark:text-gray-300 text-center">
            <strong>Evidence Based Medicine Pyramid:</strong> Visual analysis of source quality hierarchy
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
              img: EBMPyramidImage, // Use custom EBM pyramid image component
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
