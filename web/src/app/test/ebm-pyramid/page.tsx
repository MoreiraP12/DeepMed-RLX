// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import { EBMPyramid } from "~/components/deer-flow/ebm-pyramid";

// Sample data for testing the EBM pyramid component
const sampleEBMData = {
  total_sources: 10,
  levels: {
    SYSTEMATIC_REVIEWS: {
      level_number: 1,
      level_name: "Systematic Reviews",
      display_name: "Systematic Reviews & Meta-analyses",
      count: 2,
      percentage: 20.0,
      average_score: 0.91,
      color: "#2E8B57",
      sources: [
        {
          title: "Systematic review and meta-analysis of COVID-19 treatments",
          url: "https://pubmed.ncbi.nlm.nih.gov/example1",
          domain: "pubmed.ncbi.nlm.nih.gov",
          score: 0.92,
          favicon_url: "https://pubmed.ncbi.nlm.nih.gov/favicon.ico",
          content_preview: "This systematic review and meta-analysis examines the effectiveness of various COVID-19 treatments based on randomized controlled trials..."
        },
        {
          title: "Cochrane Review: Exercise therapy for chronic low back pain",
          url: "https://cochranelibrary.com/cdsr/doi/10.1002/example",
          domain: "cochranelibrary.com",
          score: 0.90,
          favicon_url: "https://cochranelibrary.com/favicon.ico",
          content_preview: "This Cochrane systematic review examines the evidence for exercise therapy in treating chronic low back pain..."
        }
      ]
    },
    GUIDELINES: {
      level_number: 2,
      level_name: "Guidelines",
      display_name: "Clinical Guidelines & CATs",
      count: 2,
      percentage: 20.0,
      average_score: 0.88,
      color: "#4682B4",
      sources: [
        {
          title: "NICE Clinical Guidelines for Diabetes Management 2024",
          url: "https://nice.org.uk/guidance/ng28",
          domain: "nice.org.uk",
          score: 0.89,
          favicon_url: "https://nice.org.uk/favicon.ico",
          content_preview: "Updated clinical practice guidelines for the management of type 2 diabetes in adults, providing evidence-based recommendations..."
        },
        {
          title: "WHO Guidelines for Global Health Emergency Preparedness",
          url: "https://who.int/publications/guidelines/emergency-preparedness",
          domain: "who.int",
          score: 0.87,
          favicon_url: "https://who.int/favicon.ico",
          content_preview: "World Health Organization guidelines for strengthening health systems' preparedness for global health emergencies..."
        }
      ]
    },
    RCTS: {
      level_number: 3,
      level_name: "Rcts",
      display_name: "Randomized Controlled Trials",
      count: 3,
      percentage: 30.0,
      average_score: 0.84,
      color: "#1E90FF",
      sources: [
        {
          title: "Randomized controlled trial of new antihypertensive drug",
          url: "https://nejm.org/doi/full/10.1056/example",
          domain: "nejm.org",
          score: 0.85,
          favicon_url: "https://nejm.org/favicon.ico",
          content_preview: "A double-blind, placebo-controlled randomized trial evaluating the efficacy and safety of a novel antihypertensive medication..."
        },
        {
          title: "Multi-center randomized trial of cardiac surgery techniques",
          url: "https://bmj.com/content/example",
          domain: "bmj.com",
          score: 0.83,
          favicon_url: "https://bmj.com/favicon.ico",
          content_preview: "A large multi-center randomized controlled trial comparing minimally invasive vs traditional cardiac surgery approaches..."
        }
      ]
    },
    COHORT: {
      level_number: 4,
      level_name: "Cohort",
      display_name: "Cohort Studies",
      count: 1,
      percentage: 10.0,
      average_score: 0.78,
      color: "#FFD700",
      sources: [
        {
          title: "Prospective cohort study of dietary factors and heart disease",
          url: "https://academic.oup.com/ajcn/example",
          domain: "academic.oup.com",
          score: 0.78,
          favicon_url: "https://academic.oup.com/favicon.ico",
          content_preview: "A 20-year prospective cohort study following 50,000 participants to assess dietary risk factors for cardiovascular disease..."
        }
      ]
    },
    CASE_CONTROL: {
      level_number: 5,
      level_name: "Case Control",
      display_name: "Case-Control Studies",
      count: 1,
      percentage: 10.0,
      average_score: 0.71,
      color: "#FF8C00",
      sources: [
        {
          title: "Case-control study of lung cancer risk factors",
          url: "https://pubmed.ncbi.nlm.nih.gov/example2",
          domain: "pubmed.ncbi.nlm.nih.gov",
          score: 0.71,
          favicon_url: "https://pubmed.ncbi.nlm.nih.gov/favicon.ico",
          content_preview: "A case-control study examining environmental and genetic risk factors for lung cancer in non-smokers..."
        }
      ]
    },
    CASE_SERIES: {
      level_number: 6,
      level_name: "Case Series",
      display_name: "Case Series & Reports",
      count: 1,
      percentage: 10.0,
      average_score: 0.68,
      color: "#FF6347",
      sources: [
        {
          title: "Case series: Rare cardiac complications in COVID-19",
          url: "https://journals.lww.com/example",
          domain: "journals.lww.com",
          score: 0.68,
          favicon_url: "https://journals.lww.com/favicon.ico",
          content_preview: "A case series describing 15 patients with rare cardiac complications following COVID-19 infection..."
        }
      ]
    }
  },
  domain_distribution: {
    "pubmed.ncbi.nlm.nih.gov": 2,
    "cochranelibrary.com": 1,
    "nice.org.uk": 1,
    "who.int": 1,
    "nejm.org": 1,
    "bmj.com": 1,
    "academic.oup.com": 1,
    "journals.lww.com": 1
  },
  top_domains: [
    ["pubmed.ncbi.nlm.nih.gov", 2],
    ["cochranelibrary.com", 1],
    ["nice.org.uk", 1],
    ["who.int", 1],
    ["nejm.org", 1]
  ],
  quality_distribution: {
    high_quality: 7,
    medium_quality: 2,
    low_quality: 1
  }
};

export default function EBMPyramidTestPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-8">
        {/* Page Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4">EBM Pyramid Test Page</h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Interactive demonstration of the Enhanced Evidence-Based Medicine pyramid component 
            with Tavily integration, featuring beautiful React visualization with logos and quality metrics.
          </p>
        </div>

        {/* EBM Pyramid Component */}
        <EBMPyramid data={sampleEBMData} />

        {/* Additional Information */}
        <div className="mt-12 max-w-4xl mx-auto">
          <div className="bg-accent/50 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4">About This Visualization</h3>
            <div className="grid md:grid-cols-2 gap-6 text-sm">
              <div>
                <h4 className="font-medium mb-2">Features:</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Interactive pyramid levels with expandable details</li>
                  <li>• Tavily source logos and favicons</li>
                  <li>• Relevance score integration</li>
                  <li>• Domain-based source classification</li>
                  <li>• Animated transitions and hover effects</li>
                  <li>• Responsive design for all devices</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">Evidence Levels:</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Level 1: Systematic Reviews & Meta-analyses</li>
                  <li>• Level 2: Clinical Guidelines & CATs</li>
                  <li>• Level 3: Randomized Controlled Trials</li>
                  <li>• Level 4: Cohort Studies</li>
                  <li>• Level 5: Case-Control Studies</li>
                  <li>• Level 6: Case Series & Reports</li>
                  <li>• Level 7: Expert Opinion</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 