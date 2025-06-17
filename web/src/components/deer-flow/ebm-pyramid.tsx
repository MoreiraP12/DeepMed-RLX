// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  BarChart3, 
  Users, 
  FileText, 
  Eye, 
  Info,
  Award,
  Shield,
  Star,
  ExternalLink,
  ChevronDown,
  ChevronUp
} from 'lucide-react';

import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Separator } from "~/components/ui/separator";
import { Tooltip } from "~/components/deer-flow/tooltip";
import { cn } from "~/lib/utils";

// Type definitions for EBM pyramid data
interface EBMSource {
  title: string;
  url: string;
  domain: string;
  score: number;
  favicon_url: string;
  content_preview: string;
}

interface EBMLevelData {
  level_number: number;
  level_name: string;
  display_name: string;
  count: number;
  percentage: number;
  average_score: number;
  color: string;
  sources: EBMSource[];
}

interface EBMPyramidData {
  total_sources: number;
  levels: Record<string, EBMLevelData>;
  domain_distribution: Record<string, number>;
  top_domains: [string, number][];
  quality_distribution: {
    high_quality: number;
    medium_quality: number;
    low_quality: number;
  };
}

// Icons for different evidence levels
const getLevelIcon = (levelName: string) => {
  switch (levelName) {
    case 'SYSTEMATIC_REVIEWS':
      return Award;
    case 'GUIDELINES':
      return Shield;
    case 'RCTS':
      return Star;
    case 'COHORT':
      return TrendingUp;
    case 'CASE_CONTROL':
      return BarChart3;
    case 'CASE_SERIES':
      return FileText;
    case 'EXPERT_OPINION':
      return Users;
    default:
      return Info;
  }
};

// Favicon component with fallback
const FaviconImage: React.FC<{ url: string; domain: string; className?: string }> = ({ 
  url, 
  domain, 
  className 
}) => {
  const [imgSrc, setImgSrc] = useState(url);
  const [hasError, setHasError] = useState(false);

  const handleError = () => {
    setHasError(true);
    // Try domain/favicon.ico as fallback
    if (!imgSrc.includes('/favicon.ico')) {
      setImgSrc(`https://${domain}/favicon.ico`);
    }
  };

  if (hasError) {
    return (
      <div className={cn("bg-accent rounded-full flex items-center justify-center", className)}>
        <ExternalLink className="w-3 h-3 text-muted-foreground" />
      </div>
    );
  }

  return (
    <img
      src={imgSrc}
      alt={`${domain} favicon`}
      className={cn("rounded-full object-cover", className)}
      onError={handleError}
      width={16}
      height={16}
    />
  );
};

// Individual pyramid level component
const PyramidLevel: React.FC<{
  levelData: EBMLevelData;
  levelKey: string;
  totalSources: number;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
}> = ({ levelData, levelKey, totalSources, index, isExpanded, onToggle }) => {
  const LevelIcon = getLevelIcon(levelKey);
  const width = Math.max(30, Math.min(100, 20 + (levelData.count / totalSources) * 80));
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="relative"
    >
      {/* Pyramid Level */}
      <div className="flex flex-col items-center mb-4">
        <motion.div
          className="relative cursor-pointer group"
          style={{ width: `${width}%` }}
          onClick={onToggle}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <div
            className="h-16 rounded-lg border-2 border-white/20 shadow-lg flex items-center justify-center relative overflow-hidden"
            style={{ 
              backgroundColor: levelData.color,
              opacity: 0.8 + (levelData.average_score * 0.2)
            }}
          >
            {/* Background pattern */}
            <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent" />
            
            {/* Content */}
            <div className="relative z-10 text-center text-white px-4">
              <div className="flex items-center justify-center gap-2 mb-1">
                <LevelIcon className="w-4 h-4" />
                <span className="font-semibold text-sm">{levelData.display_name}</span>
              </div>
              <div className="text-xs opacity-90">
                {levelData.count} sources ({levelData.percentage.toFixed(1)}%)
              </div>
              {levelData.average_score > 0 && (
                <div className="text-xs font-medium mt-1">
                  Score: {levelData.average_score.toFixed(2)}
                </div>
              )}
            </div>
            
            {/* Expand indicator */}
            <div className="absolute bottom-1 right-1">
              {isExpanded ? (
                <ChevronUp className="w-3 h-3 text-white/70" />
              ) : (
                <ChevronDown className="w-3 h-3 text-white/70" />
              )}
            </div>
            
            {/* Top source favicons */}
            {levelData.sources.length > 0 && (
              <div className="absolute top-1 left-1 flex gap-1">
                {levelData.sources.slice(0, 3).map((source, idx) => (
                  <FaviconImage
                    key={idx}
                    url={source.favicon_url}
                    domain={source.domain}
                    className="w-4 h-4 border border-white/30"
                  />
                ))}
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Expanded source details */}
      <AnimatePresence>
        {isExpanded && levelData.sources.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="mb-6"
          >
            <Card className="mx-auto max-w-2xl">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <LevelIcon className="w-5 h-5" style={{ color: levelData.color }} />
                  {levelData.display_name} Sources
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {levelData.sources.map((source, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="flex items-start gap-3 p-3 rounded-lg bg-accent/50 hover:bg-accent/70 transition-colors"
                  >
                    <FaviconImage
                      url={source.favicon_url}
                      domain={source.domain}
                      className="w-6 h-6 mt-1 border border-border"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2">
                        <h4 className="font-medium text-sm leading-5 line-clamp-2">
                          <a 
                            href={source.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="hover:text-primary transition-colors"
                          >
                            {source.title}
                          </a>
                        </h4>
                        <Badge variant="secondary" className="text-xs shrink-0">
                          {source.score.toFixed(2)}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                        {source.content_preview}
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        <Badge variant="outline" className="text-xs">
                          {source.domain}
                        </Badge>
                        <ExternalLink className="w-3 h-3 text-muted-foreground" />
                      </div>
                    </div>
                  </motion.div>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

// Statistics panel component
const StatisticsPanel: React.FC<{ data: EBMPyramidData }> = ({ data }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
      {/* Total Sources */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <FileText className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Sources</p>
              <p className="text-2xl font-bold">{data.total_sources}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quality Distribution */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-100 dark:bg-emerald-900/20">
              <Award className="w-5 h-5 text-emerald-600" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">High Quality</p>
              <p className="text-2xl font-bold text-emerald-600">
                {data.quality_distribution.high_quality}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Top Domain */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/20">
              <TrendingUp className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Top Domain</p>
              <p className="text-sm font-semibold truncate">
                {data.top_domains[0]?.[0] || 'N/A'}
              </p>
              <p className="text-xs text-muted-foreground">
                {data.top_domains[0]?.[1] || 0} sources
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Domain distribution component
const DomainDistribution: React.FC<{ domains: [string, number][] }> = ({ domains }) => {
  const maxCount = Math.max(...domains.map(([, count]) => count));
  
  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          Top Source Domains
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {domains.map(([domain, count], idx) => (
            <motion.div
              key={domain}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="flex items-center gap-3"
            >
              <div className="w-32 text-sm font-medium truncate">{domain}</div>
              <div className="flex-1 bg-accent rounded-full h-2 overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-primary to-primary/60"
                  initial={{ width: 0 }}
                  animate={{ width: `${(count / maxCount) * 100}%` }}
                  transition={{ delay: idx * 0.1 + 0.3, duration: 0.6 }}
                />
              </div>
              <Badge variant="secondary" className="text-xs">
                {count}
              </Badge>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

// Main EBM Pyramid component
export const EBMPyramid: React.FC<{
  data?: EBMPyramidData;
  jsonPath?: string;
  className?: string;
}> = ({ data: propData, jsonPath, className }) => {
  const [data, setData] = useState<EBMPyramidData | null>(propData || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedLevels, setExpandedLevels] = useState<Set<string>>(new Set());

  // Load data from JSON if path provided
  useEffect(() => {
    if (jsonPath && !propData) {
      setLoading(true);
      fetch(jsonPath)
        .then(res => res.json())
        .then(setData)
        .catch(err => setError(err.message))
        .finally(() => setLoading(false));
    }
  }, [jsonPath, propData]);

  const toggleLevel = (levelKey: string) => {
    const newExpanded = new Set(expandedLevels);
    if (newExpanded.has(levelKey)) {
      newExpanded.delete(levelKey);
    } else {
      newExpanded.add(levelKey);
    }
    setExpandedLevels(newExpanded);
  };

  if (loading) {
    return (
      <div className={cn("flex items-center justify-center p-8", className)}>
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn("text-center p-8 text-destructive", className)}>
        <p>Error loading EBM pyramid data: {error}</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className={cn("text-center p-8 text-muted-foreground", className)}>
        <p>No EBM pyramid data available</p>
      </div>
    );
  }

  // Sort levels by evidence quality (highest first)
  const sortedLevels = Object.entries(data.levels).sort(
    ([, a], [, b]) => a.level_number - b.level_number
  );

  return (
    <div className={cn("max-w-6xl mx-auto p-6", className)}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <div className="flex items-center justify-center gap-2 mb-4">
          <div className="p-3 rounded-full bg-gradient-to-r from-emerald-500 to-blue-500">
            <Award className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-3xl font-bold bg-gradient-to-r from-emerald-600 to-blue-600 bg-clip-text text-transparent">
            Evidence-Based Medicine Pyramid
          </h2>
        </div>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Interactive visualization of evidence quality hierarchy with Tavily source analysis, 
          relevance scores, and domain intelligence.
        </p>
      </motion.div>

      {/* Statistics */}
      <StatisticsPanel data={data} />

      {/* Pyramid Visualization */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="text-center">Evidence Quality Hierarchy</CardTitle>
          <p className="text-center text-sm text-muted-foreground">
            Click on any level to view detailed source information
          </p>
        </CardHeader>
        <CardContent className="py-8">
          <div className="space-y-6">
            {sortedLevels.map(([levelKey, levelData], index) => (
              <PyramidLevel
                key={levelKey}
                levelData={levelData}
                levelKey={levelKey}
                totalSources={data.total_sources}
                index={index}
                isExpanded={expandedLevels.has(levelKey)}
                onToggle={() => toggleLevel(levelKey)}
              />
            ))}
          </div>
          
          {/* Quality scale indicator */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="mt-8 flex items-center justify-between text-sm text-muted-foreground"
          >
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-emerald-600" />
              <span>Highest Quality</span>
            </div>
            <div className="flex-1 mx-4 h-px bg-gradient-to-r from-emerald-600 via-yellow-500 to-red-600"></div>
            <div className="flex items-center gap-2">
              <span>Lowest Quality</span>
              <TrendingUp className="w-4 h-4 text-red-600 rotate-180" />
            </div>
          </motion.div>
        </CardContent>
      </Card>

      {/* Domain Distribution */}
      {data.top_domains.length > 0 && (
        <DomainDistribution domains={data.top_domains} />
      )}

      {/* Quality Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Eye className="w-5 h-5" />
            Evidence Quality Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="p-4 rounded-lg bg-emerald-50 dark:bg-emerald-900/20">
              <div className="text-2xl font-bold text-emerald-600">
                {data.quality_distribution.high_quality}
              </div>
              <div className="text-sm text-muted-foreground">High Quality</div>
              <div className="text-xs text-muted-foreground">
                Levels 1-3 (Reviews, Guidelines, RCTs)
              </div>
            </div>
            <div className="p-4 rounded-lg bg-yellow-50 dark:bg-yellow-900/20">
              <div className="text-2xl font-bold text-yellow-600">
                {data.quality_distribution.medium_quality}
              </div>
              <div className="text-sm text-muted-foreground">Medium Quality</div>
              <div className="text-xs text-muted-foreground">
                Levels 4-5 (Observational Studies)
              </div>
            </div>
            <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20">
              <div className="text-2xl font-bold text-red-600">
                {data.quality_distribution.low_quality}
              </div>
              <div className="text-sm text-muted-foreground">Low Quality</div>
              <div className="text-xs text-muted-foreground">
                Levels 6-7 (Case Studies, Opinion)
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}; 