// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { useState, useEffect } from "react";
import { useMessage } from "~/core/store";
import { MarkdownWithThinking } from "~/components/deer-flow/markdown-with-thinking";

interface TavilySource {
  type: string;
  title: string;
  url: string;
  score?: number;
  content?: string;
}

interface TavilySourcesSectionProps {
  messageId: string;
}

export function TavilySourcesSection({ messageId }: TavilySourcesSectionProps) {
  const [sourcesMarkdown, setSourcesMarkdown] = useState<string>("");
  const [hoveredSource, setHoveredSource] = useState<TavilySource | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const message = useMessage(messageId);

  // Define pyramid levels with evidence hierarchy
  const pyramidLevels = [
    {
      id: 1,
      title: "Systematic Reviews & Meta-Analyses",
      shortTitle: "Systematic Reviews",
      minScore: 0.9,
      color: "#0f4878",
      width: 20, // percentage of base width
      sources: [] as TavilySource[]
    },
    {
      id: 2,
      title: "Randomized Controlled Trials",
      shortTitle: "RCTs",
      minScore: 0.8,
      color: "#276aa0", 
      width: 35,
      sources: [] as TavilySource[]
    },
    {
      id: 3,
      title: "Cohort Studies",
      shortTitle: "Cohort Studies", 
      minScore: 0.7,
      color: "#3d8cca",
      width: 50,
      sources: [] as TavilySource[]
    },
    {
      id: 4,
      title: "Case-Control Studies",
      shortTitle: "Case-Control",
      minScore: 0.5,
      color: "#70acdf",
      width: 65,
      sources: [] as TavilySource[]
    },
    {
      id: 5,
      title: "Case Series & Reports", 
      shortTitle: "Case Reports",
      minScore: 0.3,
      color: "#9fc8ef",
      width: 80,
      sources: [] as TavilySource[]
    },
    {
      id: 6,
      title: "Background Info & Expert Opinion",
      shortTitle: "Expert Opinion",
      minScore: 0.0,
      color: "#cfe3f8",
      width: 100, // base width
      sources: [] as TavilySource[]
    }
  ];

  useEffect(() => {
    // Get tavily sources from the message
    const tavilySources = message?.tavilySources || [];
    
    console.log("TavilySourcesSection - Message:", message);
    console.log("TavilySourcesSection - Sources:", tavilySources);
    
    // Generate sources markdown if we have sources
    if (tavilySources.length > 0) {
      const markdown = generateSourcesMarkdown(tavilySources);
      console.log("TavilySourcesSection - Generated markdown:", markdown);
      setSourcesMarkdown(markdown);
    } else {
      setSourcesMarkdown("");
      console.log("TavilySourcesSection - No sources found");
    }
  }, [message?.tavilySources]);

  // Categorize sources into pyramid levels
  const categorizedLevels = pyramidLevels.map(level => ({
    ...level,
    sources: (message?.tavilySources || []).filter((source: TavilySource) => {
      const score = source.score || 0;
      return score >= level.minScore && 
             !pyramidLevels.find(l => l.minScore > level.minScore && score >= l.minScore);
    })
  }));

  const extractDomain = (url: string): string => {
    try {
      const parsedUrl = new URL(url);
      return parsedUrl.hostname;
    } catch {
      return "unknown";
    }
  };

  const generateSourcesMarkdown = (sources: TavilySource[]): string => {
    if (!sources || sources.length === 0) return "";

    let markdown = "## 📋 Detailed Source Information\n\n";
    
    sources.forEach((source, index) => {
      const domain = extractDomain(source.url);
      const faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&sz=16`;
      const qualityIndicator = getQualityIndicator(source.score || 0);
      
      markdown += `### ${index + 1}. ![${domain} favicon](${faviconUrl}) [${source.title}](${source.url})\n\n`;
      markdown += `**Quality Score:** ${qualityIndicator} (${((source.score || 0) * 100).toFixed(0)}%)\n\n`;
      if (source.content) {
        const preview = source.content.substring(0, 150) + "...";
        markdown += `**Preview:** ${preview}\n\n`;
      }
    });

    return markdown;
  };

  const getQualityIndicator = (score: number): string => {
    if (score >= 0.9) return "🟢 Excellent";
    if (score >= 0.8) return "🟡 Good";
    if (score >= 0.7) return "🟠 Fair";
    return "🔴 Poor";
  };

  // Only render if we have sources
  if (!sourcesMarkdown) {
    return null;
  }

  return (
    <div className="mt-8 pt-4 border-t border-border/40">
      {/* Global Tooltip */}
      {hoveredSource && (
        <div 
          className="fixed px-4 py-3 bg-gray-900 text-white text-sm rounded-lg shadow-2xl border border-gray-700 pointer-events-none"
          style={{ 
            left: mousePosition.x + 10,
            top: mousePosition.y - 80,
            zIndex: 10000,
            maxWidth: '300px'
          }}
        >
          <div className="font-semibold mb-1">{hoveredSource.title}</div>
          <div className="text-gray-300 text-xs mb-1">{extractDomain(hoveredSource.url)}</div>
          <div className="text-blue-300 text-xs">Quality Score: {Math.round((hoveredSource.score || 0) * 100)}%</div>
          {/* Tooltip arrow */}
          <div className="absolute top-full left-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
        </div>
      )}
      
      {/* Evidence-Based Medicine Pyramid */}
      <div className="mb-8 flex justify-center">
        <div className="relative max-w-2xl w-full">
          {/* Title */}
          <div className="text-center mb-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">
              Evidence-Based Medicine Pyramid
            </h3>
            <p className="text-sm text-gray-600">
              Sources organized by evidence quality and research methodology
            </p>
          </div>

                     {/* Main pyramid container */}
           <div className="relative">

                         {/* Pyramid levels */}
             <div className="relative" style={{ width: '100%', height: '360px' }}>
               {categorizedLevels.map((level, index) => {
                 // Calculate proper pyramid geometry
                 const totalLevels = categorizedLevels.length;
                 const levelHeight = 60;
                 const baseWidth = 400; // Base width in pixels (bottom level)
                 const topWidth = 0;    // Top width = 0 for perfect triangle point
                 
                 // Calculate width for this level (linear progression from point to wide bottom)
                 const levelWidth = topWidth + ((baseWidth - topWidth) * index / (totalLevels - 1));
                 const nextLevelWidth = index < totalLevels - 1 
                   ? topWidth + ((baseWidth - topWidth) * (index + 1) / (totalLevels - 1))
                   : baseWidth;
                 
                 const yPosition = index * levelHeight;
                 
                 // Calculate trapezoid points for perfect pyramid shape
                 let clipPath;
                 
                 if (index === 0) {
                   // First level: Triangle point at top
                   const bottomLeft = (400 - nextLevelWidth) / 2;
                   const bottomRight = bottomLeft + nextLevelWidth;
                   clipPath = `polygon(50% 0%, 50% 0%, ${(bottomRight/400)*100}% 100%, ${(bottomLeft/400)*100}% 100%)`;
                 } else if (index === totalLevels - 1) {
                   // Last level: Trapezoid that expands outward at bottom
                   const topLeft = (400 - levelWidth) / 2;
                   const topRight = topLeft + levelWidth;
                   const bottomExpansion = 30; // How much to expand outward at bottom
                   const bottomLeft = Math.max(0, topLeft - bottomExpansion);
                   const bottomRight = Math.min(400, topRight + bottomExpansion);
                   clipPath = `polygon(${(topLeft/400)*100}% 0%, ${(topRight/400)*100}% 0%, ${(bottomRight/400)*100}% 100%, ${(bottomLeft/400)*100}% 100%)`;
                 } else {
                   // Middle levels: Regular trapezoids
                   const topLeft = (400 - levelWidth) / 2;
                   const topRight = topLeft + levelWidth;
                   const bottomLeft = (400 - nextLevelWidth) / 2;
                   const bottomRight = bottomLeft + nextLevelWidth;
                   clipPath = `polygon(${(topLeft/400)*100}% 0%, ${(topRight/400)*100}% 0%, ${(bottomRight/400)*100}% 100%, ${(bottomLeft/400)*100}% 100%)`;
                 }
                 
                 return (
                   <div
                     key={level.id}
                     className="absolute transition-all duration-300 hover:brightness-110"
                     style={{
                       left: '50%',
                       top: `${yPosition}px`,
                       transform: 'translateX(-50%)',
                       width: '400px',
                       height: `${levelHeight}px`,
                       background: level.sources.length > 0 
                         ? `linear-gradient(135deg, ${level.color}, ${level.color}cc)`
                         : `linear-gradient(135deg, #e5e7eb, #d1d5db)`,
                       opacity: level.sources.length > 0 ? 1 : 0.4,
                       clipPath: clipPath,
                       border: `1px solid ${level.sources.length > 0 ? level.color : '#9ca3af'}`,
                       zIndex: totalLevels - index,
                     }}
                                        >
                       <div className="flex items-center justify-center h-full">
                         {/* Special layout for top triangle level */}
                         {index === 0 ? (
                           <div className="flex flex-col items-center gap-1" style={{ marginTop: '10px' }}>
                             {/* Level title for top level */}
                             <div className="text-center">
                               <div 
                                 className="font-semibold text-xs"
                                 style={{ 
                                   color: level.sources.length > 0 ? 'white' : '#6b7280',
                                   textShadow: level.sources.length > 0 ? '1px 1px 2px rgba(0,0,0,0.7)' : 'none'
                                 }}
                               >
                                 {level.shortTitle}
                               </div>
                             </div>
                             
                             {/* Favicons for top level */}
                             {level.sources.length > 0 && (
                               <div className="flex items-center gap-1 justify-center">
                                 {level.sources.map((source, sourceIndex) => {
                                   const domain = extractDomain(source.url);
                                   const faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&sz=14`;
                                   
                                   return (
                                     <div
                                       key={sourceIndex}
                                       className="relative cursor-pointer transform transition-transform hover:scale-125"
                                       onClick={() => window.open(source.url, '_blank')}
                                       onMouseEnter={(e) => {
                                         setHoveredSource(source);
                                         setMousePosition({ x: e.clientX, y: e.clientY });
                                       }}
                                       onMouseLeave={() => {
                                         setHoveredSource(null);
                                       }}
                                       onMouseMove={(e) => {
                                         if (hoveredSource === source) {
                                           setMousePosition({ x: e.clientX, y: e.clientY });
                                         }
                                       }}
                                     >
                                       <div className="w-4 h-4 bg-white rounded-full p-0.5 shadow-lg border border-white/50">
                                         <img
                                           src={faviconUrl}
                                           alt={domain}
                                           className="w-full h-full object-contain rounded-full"
                                           onError={(e) => {
                                             e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2IiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik04IDJMMTQgNi41VjEzSDJWNi41TDggMloiIGZpbGw9IiM5Q0EzQUYiLz4KPC9zdmc+';
                                           }}
                                         />
                                       </div>
                                     </div>
                                   );
                                 })}
                               </div>
                             )}
                             
                             {/* No sources indicator for top level */}
                             {level.sources.length === 0 && (
                               <div className="text-xs text-gray-500 italic text-center">
                                 No sources
                               </div>
                             )}
                           </div>
                         ) : (
                           /* Regular layout for other levels */
                           <div className="flex items-center gap-4" style={{
                             maxWidth: `${Math.min(levelWidth, nextLevelWidth) - 40}px`
                           }}>
                             {/* Level title */}
                             <div className="flex-shrink-0">
                               <div 
                                 className="font-semibold text-sm"
                                 style={{ 
                                   color: level.sources.length > 0 ? 'white' : '#6b7280',
                                   textShadow: level.sources.length > 0 ? '1px 1px 2px rgba(0,0,0,0.7)' : 'none'
                                 }}
                               >
                                 {level.shortTitle}
                               </div>
                             </div>

                             {/* Favicons */}
                             {level.sources.length > 0 && (
                               <div className="flex items-center gap-2 flex-wrap">
                                 {level.sources.map((source, sourceIndex) => {
                                   const domain = extractDomain(source.url);
                                   const faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&sz=16`;
                                   
                                   return (
                                     <div
                                       key={sourceIndex}
                                       className="relative cursor-pointer transform transition-transform hover:scale-125"
                                       onClick={() => window.open(source.url, '_blank')}
                                       onMouseEnter={(e) => {
                                         setHoveredSource(source);
                                         setMousePosition({ x: e.clientX, y: e.clientY });
                                       }}
                                       onMouseLeave={() => {
                                         setHoveredSource(null);
                                       }}
                                       onMouseMove={(e) => {
                                         if (hoveredSource === source) {
                                           setMousePosition({ x: e.clientX, y: e.clientY });
                                         }
                                       }}
                                     >
                                       <div className="w-5 h-5 bg-white rounded-full p-0.5 shadow-lg border border-white/50">
                                         <img
                                           src={faviconUrl}
                                           alt={domain}
                                           className="w-full h-full object-contain rounded-full"
                                           onError={(e) => {
                                             e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2IiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik04IDJMMTQgNi41VjEzSDJWNi41TDggMloiIGZpbGw9IiM5Q0EzQUYiLz4KPC9zdmc+';
                                           }}
                                         />
                                       </div>
                                     </div>
                                   );
                                 })}
                               </div>
                             )}
                             
                             {/* No sources indicator */}
                             {level.sources.length === 0 && (
                               <div className="text-xs text-gray-500 italic ml-2">
                                 No sources
                               </div>
                             )}
                           </div>
                         )}
                       </div>
                     </div>
                   );
                                  })}
               </div>
           </div>

                   </div>
       </div>
    </div>
  );
} 