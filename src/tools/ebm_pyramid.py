# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import re
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from PIL import Image as PILImage
import io

logger = logging.getLogger(__name__)

class EvidenceLevel(Enum):
    """Evidence levels in EBM pyramid from highest to lowest quality"""
    SYSTEMATIC_REVIEWS = 1  # Systematic Reviews & Meta-analyses
    GUIDELINES = 2          # Clinical Guidelines & Critically Appraised Topics  
    RCTS = 3               # Randomized Controlled Trials
    COHORT = 4             # Cohort Studies
    CASE_CONTROL = 5       # Case-Control Studies
    CASE_SERIES = 6        # Case Series & Reports
    EXPERT_OPINION = 7     # Expert Opinion

@dataclass
class TavilySourceData:
    """Enhanced source data from Tavily with rich metadata"""
    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = None
    favicon_url: Optional[str] = None
    domain: Optional[str] = None
    evidence_level: Optional[EvidenceLevel] = None
    
    def __post_init__(self):
        if self.url:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc
            # Generate favicon URL from domain
            self.favicon_url = f"https://{self.domain}/favicon.ico"

@dataclass
class EBMPyramidData:
    """Enhanced EBM pyramid data with Tavily integration"""
    sources_by_level: Dict[EvidenceLevel, List[TavilySourceData]] = field(default_factory=dict)
    total_sources: int = 0
    average_score_by_level: Dict[EvidenceLevel, float] = field(default_factory=dict)
    domain_distribution: Dict[str, int] = field(default_factory=dict)
    
    def add_source(self, source: TavilySourceData):
        """Add a source to the appropriate evidence level"""
        if source.evidence_level:
            if source.evidence_level not in self.sources_by_level:
                self.sources_by_level[source.evidence_level] = []
            self.sources_by_level[source.evidence_level].append(source)
            self.total_sources += 1
            
            # Update domain distribution
            if source.domain:
                self.domain_distribution[source.domain] = self.domain_distribution.get(source.domain, 0) + 1
    
    def calculate_average_scores(self):
        """Calculate average Tavily scores by evidence level"""
        for level, sources in self.sources_by_level.items():
            if sources:
                self.average_score_by_level[level] = sum(s.score for s in sources) / len(sources)

class EnhancedEBMSourceClassifier:
    """Enhanced classifier that uses Tavily metadata for better source classification"""
    
    def __init__(self):
        # Domain-based classification with confidence weights
        self.domain_patterns = {
            EvidenceLevel.SYSTEMATIC_REVIEWS: {
                'cochrane': 0.95,
                'pubmed': 0.8,  # PubMed can have various study types
                'systematicreview': 0.9,
                'bmj.com': 0.85,
                'nejm.org': 0.85,
                'thelancet.com': 0.85,
            },
            EvidenceLevel.GUIDELINES: {
                'nice.org.uk': 0.95,
                'who.int': 0.9,
                'cdc.gov': 0.9,
                'uptodate.com': 0.9,
                'guidelines.co.uk': 0.95,
                'acp.org': 0.85,
                'aha.org': 0.85,
            },
            EvidenceLevel.RCTS: {
                'clinicaltrials.gov': 0.9,
                'pubmed': 0.7,  # Needs content analysis
                'nejm.org': 0.8,
                'bmj.com': 0.8,
                'jama.jamanetwork.com': 0.85,
            },
            EvidenceLevel.COHORT: {
                'pubmed': 0.6,
                'academic.oup.com': 0.7,
                'springer.com': 0.7,
            },
            EvidenceLevel.CASE_CONTROL: {
                'pubmed': 0.5,
                'sciencedirect.com': 0.6,
            },
            EvidenceLevel.CASE_SERIES: {
                'pubmed': 0.4,
                'journals.lww.com': 0.6,
            },
            EvidenceLevel.EXPERT_OPINION: {
                'medscape.com': 0.7,
                'webmd.com': 0.6,
                'mayoclinic.org': 0.8,
                'healthline.com': 0.6,
            }
        }
        
        # Content-based classification patterns
        self.content_patterns = {
            EvidenceLevel.SYSTEMATIC_REVIEWS: [
                r'systematic review',
                r'meta-analysis',
                r'cochrane review',
                r'systematic literature review',
                r'pooled analysis'
            ],
            EvidenceLevel.GUIDELINES: [
                r'clinical guideline',
                r'practice guideline',
                r'consensus statement',
                r'clinical practice',
                r'recommendation'
            ],
            EvidenceLevel.RCTS: [
                r'randomized controlled trial',
                r'randomised controlled trial',
                r'RCT',
                r'double-blind',
                r'placebo-controlled'
            ],
            EvidenceLevel.COHORT: [
                r'cohort study',
                r'longitudinal study',
                r'prospective study',
                r'follow-up study'
            ],
            EvidenceLevel.CASE_CONTROL: [
                r'case-control study',
                r'case control study',
                r'retrospective study'
            ],
            EvidenceLevel.CASE_SERIES: [
                r'case series',
                r'case report',
                r'case study'
            ]
        }
    
    def classify_tavily_source(self, tavily_result: Dict[str, Any]) -> TavilySourceData:
        """Classify a Tavily search result into evidence levels"""
        source = TavilySourceData(
            title=tavily_result.get('title', ''),
            url=tavily_result.get('url', ''),
            content=tavily_result.get('content', ''),
            score=tavily_result.get('score', 0.0),
            raw_content=tavily_result.get('raw_content')
        )
        
        # Determine evidence level
        source.evidence_level = self._determine_evidence_level(source)
        
        return source
    
    def _determine_evidence_level(self, source: TavilySourceData) -> EvidenceLevel:
        """Determine evidence level using domain and content analysis"""
        best_level = EvidenceLevel.EXPERT_OPINION
        best_confidence = 0.0
        
        # Domain-based classification
        if source.domain:
            for level, domains in self.domain_patterns.items():
                for domain_pattern, confidence in domains.items():
                    if domain_pattern in source.domain.lower():
                        if confidence > best_confidence:
                            best_level = level
                            best_confidence = confidence
        
        # Content-based classification (can override domain)
        content_text = f"{source.title} {source.content}"
        if source.raw_content:
            content_text += f" {source.raw_content}"
        
        content_text = content_text.lower()
        
        for level, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_text, re.IGNORECASE):
                    # Content patterns have high confidence
                    content_confidence = 0.8
                    if content_confidence > best_confidence:
                        best_level = level
                        best_confidence = content_confidence
                    break
        
        return best_level

class EnhancedEBMPyramidVisualizer:
    """Enhanced visualizer with Tavily logos and rich metadata display"""
    
    def __init__(self):
        self.colors = {
            EvidenceLevel.SYSTEMATIC_REVIEWS: '#2E8B57',    # Sea Green
            EvidenceLevel.GUIDELINES: '#4682B4',           # Steel Blue  
            EvidenceLevel.RCTS: '#1E90FF',                 # Dodger Blue
            EvidenceLevel.COHORT: '#FFD700',               # Gold
            EvidenceLevel.CASE_CONTROL: '#FF8C00',         # Dark Orange
            EvidenceLevel.CASE_SERIES: '#FF6347',          # Tomato
            EvidenceLevel.EXPERT_OPINION: '#DC143C'        # Crimson
        }
        
        self.level_names = {
            EvidenceLevel.SYSTEMATIC_REVIEWS: "Systematic Reviews\n& Meta-analyses",
            EvidenceLevel.GUIDELINES: "Clinical Guidelines\n& CATs",
            EvidenceLevel.RCTS: "Randomized\nControlled Trials",
            EvidenceLevel.COHORT: "Cohort Studies",
            EvidenceLevel.CASE_CONTROL: "Case-Control\nStudies", 
            EvidenceLevel.CASE_SERIES: "Case Series\n& Reports",
            EvidenceLevel.EXPERT_OPINION: "Expert Opinion"
        }
    
    def download_favicon(self, url: str) -> Optional[PILImage.Image]:
        """Download and process favicon from URL"""
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                image = PILImage.open(io.BytesIO(response.content))
                # Convert to RGBA and resize
                if image.mode != 'RGBA':
                    image = image.convert('RGBA')
                image = image.resize((32, 32), PILImage.Resampling.LANCZOS)
                return image
        except Exception as e:
            logger.debug(f"Failed to download favicon from {url}: {e}")
        return None
    
    def create_enhanced_pyramid(self, pyramid_data: EBMPyramidData, output_path: str = "ebm_pyramid_enhanced.png"):
        """Create enhanced pyramid visualization with Tavily metadata"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # Left side: Traditional pyramid
        self._draw_pyramid(ax1, pyramid_data)
        
        # Right side: Detailed source breakdown
        self._draw_source_details(ax2, pyramid_data)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def _draw_pyramid(self, ax, pyramid_data: EBMPyramidData):
        """Draw the traditional EBM pyramid with enhancements"""
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.1, 1.4)
        ax.set_aspect('equal')
        ax.axis('off')
        
        levels = list(EvidenceLevel)
        y_positions = np.linspace(1.2, 0.1, len(levels))
        widths = np.linspace(0.3, 2.0, len(levels))
        
        for i, level in enumerate(levels):
            y = y_positions[i]
            width = widths[i]
            height = 0.12
            
            # Get source count and average score
            sources = pyramid_data.sources_by_level.get(level, [])
            source_count = len(sources)
            avg_score = pyramid_data.average_score_by_level.get(level, 0.0)
            
            # Draw pyramid segment
            color = self.colors[level]
            alpha = 0.3 + (avg_score * 0.7) if avg_score > 0 else 0.3
            
            rect = patches.Rectangle(
                (-width/2, y), width, height,
                facecolor=color, alpha=alpha,
                edgecolor='black', linewidth=1.5
            )
            ax.add_patch(rect)
            
            # Add level name and count
            level_text = self.level_names[level]
            if source_count > 0:
                level_text += f"\n({source_count} sources)"
                if avg_score > 0:
                    level_text += f"\nAvg Score: {avg_score:.2f}"
            
            ax.text(0, y + height/2, level_text, 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Add favicons for top sources
            if sources:
                self._add_favicons_to_pyramid(ax, sources[:3], y, width, height)
        
        ax.set_title('Evidence-Based Medicine Pyramid\nwith Tavily Source Analysis', 
                    fontsize=14, fontweight='bold', pad=20)
    
    def _add_favicons_to_pyramid(self, ax, sources: List[TavilySourceData], y: float, width: float, height: float):
        """Add favicon images to pyramid segments"""
        favicon_spacing = min(width / (len(sources) + 1), 0.15)
        start_x = -width/2 + favicon_spacing
        
        for i, source in enumerate(sources):
            if source.favicon_url:
                favicon = self.download_favicon(source.favicon_url)
                if favicon:
                    try:
                        # Convert PIL image to matplotlib format
                        favicon_array = np.array(favicon)
                        im = OffsetImage(favicon_array, zoom=0.5)
                        
                        x_pos = start_x + i * favicon_spacing
                        ab = AnnotationBbox(im, (x_pos, y + height + 0.02), 
                                          frameon=False, pad=0)
                        ax.add_artist(ab)
                    except Exception as e:
                        logger.debug(f"Failed to add favicon for {source.domain}: {e}")
    
    def _draw_source_details(self, ax, pyramid_data: EBMPyramidData):
        """Draw detailed source breakdown panel"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        y_pos = 9.5
        ax.text(5, y_pos, 'Source Details & Quality Metrics', 
               ha='center', va='top', fontsize=14, fontweight='bold')
        
        y_pos -= 0.8
        
        # Overall statistics
        ax.text(0.5, y_pos, f'Total Sources: {pyramid_data.total_sources}', 
               fontsize=12, fontweight='bold')
        y_pos -= 0.4
        
        # Top domains
        if pyramid_data.domain_distribution:
            top_domains = sorted(pyramid_data.domain_distribution.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
            ax.text(0.5, y_pos, 'Top Domains:', fontsize=11, fontweight='bold')
            y_pos -= 0.3
            for domain, count in top_domains:
                ax.text(1, y_pos, f'• {domain}: {count} sources', fontsize=10)
                y_pos -= 0.25
        
        y_pos -= 0.3
        
        # Evidence level breakdown
        ax.text(0.5, y_pos, 'Evidence Level Breakdown:', fontsize=11, fontweight='bold')
        y_pos -= 0.3
        
        for level in EvidenceLevel:
            sources = pyramid_data.sources_by_level.get(level, [])
            if sources:
                count = len(sources)
                avg_score = pyramid_data.average_score_by_level.get(level, 0.0)
                level_name = self.level_names[level].replace('\n', ' ')
                
                # Color-coded level indicator
                color = self.colors[level]
                ax.add_patch(patches.Rectangle((0.5, y_pos-0.08), 0.3, 0.15, 
                                             facecolor=color, alpha=0.7))
                
                ax.text(1, y_pos, f'{level_name}: {count} sources (avg: {avg_score:.2f})', 
                       fontsize=10)
                y_pos -= 0.3

def extract_tavily_sources_from_observations(observations: List[str]) -> List[Dict[str, Any]]:
    """Extract Tavily search results from observation text"""
    tavily_sources = []
    
    for obs in observations:
        # Look for JSON-like structures that contain Tavily results
        # This would typically come from the search tool outputs
        try:
            # Try to find JSON structures in the observation
            import json
            # Look for patterns that indicate Tavily results
            lines = obs.split('\n')
            for line in lines:
                if 'score' in line and 'url' in line and 'title' in line:
                    try:
                        # Try to parse as JSON
                        result = json.loads(line)
                        if isinstance(result, dict) and 'score' in result:
                            tavily_sources.append(result)
                    except:
                        continue
        except Exception as e:
            logger.debug(f"Error extracting Tavily sources: {e}")
    
    return tavily_sources

def generate_enhanced_ebm_pyramid_for_research(observations: List[str], output_dir: str = "outputs") -> Optional[str]:
    """Enhanced function to generate EBM pyramid with Tavily integration"""
    try:
        # Extract Tavily sources
        tavily_results = extract_tavily_sources_from_observations(observations)
        
        if not tavily_results:
            logger.warning("No Tavily search results found in observations")
            return None
        
        # Initialize components
        classifier = EnhancedEBMSourceClassifier()
        pyramid_data = EBMPyramidData()
        
        # Classify sources
        for result in tavily_results:
            source = classifier.classify_tavily_source(result)
            pyramid_data.add_source(source)
        
        # Calculate statistics
        pyramid_data.calculate_average_scores()
        
        if pyramid_data.total_sources == 0:
            logger.warning("No sources were classified successfully")
            return None
        
        # Create visualization
        Path(output_dir).mkdir(exist_ok=True)
        output_path = Path(output_dir) / "enhanced_ebm_pyramid.png"
        
        visualizer = EnhancedEBMPyramidVisualizer()
        result_path = visualizer.create_enhanced_pyramid(pyramid_data, str(output_path))
        
        logger.info(f"Enhanced EBM pyramid generated: {result_path}")
        logger.info(f"Total sources analyzed: {pyramid_data.total_sources}")
        
        # Generate summary report
        summary = generate_ebm_summary_report(pyramid_data)
        logger.info(f"EBM Analysis Summary:\n{summary}")
        
        return result_path
        
    except Exception as e:
        logger.error(f"Error generating enhanced EBM pyramid: {e}")
        return None

def generate_ebm_summary_report(pyramid_data: EBMPyramidData) -> str:
    """Generate a textual summary of the EBM analysis"""
    lines = []
    lines.append("=== Evidence-Based Medicine Source Analysis ===")
    lines.append(f"Total Sources Analyzed: {pyramid_data.total_sources}")
    lines.append("")
    
    lines.append("Evidence Quality Distribution:")
    for level in EvidenceLevel:
        sources = pyramid_data.sources_by_level.get(level, [])
        if sources:
            count = len(sources)
            percentage = (count / pyramid_data.total_sources) * 100
            avg_score = pyramid_data.average_score_by_level.get(level, 0.0)
            level_name = level.name.replace('_', ' ').title()
            lines.append(f"  • {level_name}: {count} sources ({percentage:.1f}%) - Avg Score: {avg_score:.2f}")
    
    lines.append("")
    lines.append("Top Source Domains:")
    if pyramid_data.domain_distribution:
        top_domains = sorted(pyramid_data.domain_distribution.items(), 
                           key=lambda x: x[1], reverse=True)[:5]
        for domain, count in top_domains:
            lines.append(f"  • {domain}: {count} sources")
    
    return "\n".join(lines)

# Backward compatibility function
def generate_ebm_pyramid_for_research(observations: List[str], output_dir: str = "outputs") -> Optional[str]:
    """Legacy function name for backward compatibility"""
    return generate_enhanced_ebm_pyramid_for_research(observations, output_dir) 