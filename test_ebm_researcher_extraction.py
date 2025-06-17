#!/usr/bin/env python3
"""
Test script to verify EBM pyramid extraction from researcher execution results

This simulates the researcher node output format containing Tavily search results
and tests whether the EBM pyramid can correctly extract and analyze them.
"""

import logging
from src.tools.ebm_pyramid import (
    extract_tavily_sources_from_observations,
    generate_enhanced_ebm_pyramid_for_research
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_researcher_tavily_extraction():
    """Test EBM pyramid extraction from simulated researcher execution results"""
    
    print("🔬 Testing EBM Pyramid Extraction from Researcher Results")
    print("=" * 60)
    
    # Simulate realistic researcher step execution results that would contain Tavily search outputs
    simulated_researcher_outputs = [
        # Research Step 1: COVID-19 Treatment Guidelines  
        """Based on my research into COVID-19 treatment effectiveness in pediatric patients, I found several key sources:

## Key Findings

The search revealed several important clinical guidelines and studies:

Tool web_search returned:
[
    {
        "type": "page",
        "title": "COVID-19 Treatment Guidelines for Pediatric Patients - NIH",
        "url": "https://www.covid19treatmentguidelines.nih.gov/management/clinical-management/hospitalized-adults--therapeutic-management/",
        "content": "The National Institutes of Health provides comprehensive guidelines for treating COVID-19 in pediatric patients. Current recommendations emphasize supportive care and evidence-based interventions.",
        "score": 0.92
    },
    {
        "type": "page", 
        "title": "Systematic Review: COVID-19 Treatments in Children - Cochrane Library",
        "url": "https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.CD013791.pub2/full",
        "content": "A systematic review and meta-analysis examining the effectiveness of various COVID-19 treatments in pediatric populations. The review analyzed 23 randomized controlled trials.",
        "score": 0.89
    },
    {
        "type": "page",
        "title": "Pediatric COVID-19 Management: Clinical Practice Guidelines",
        "url": "https://pubmed.ncbi.nlm.nih.gov/34567890/",
        "content": "Clinical practice guidelines from the American Academy of Pediatrics for managing COVID-19 in children and adolescents.",
        "score": 0.85
    }
]

These sources provide evidence-based recommendations for pediatric COVID-19 treatment, including remdesivir use in hospitalized patients aged 12 and above, and dexamethasone for those requiring respiratory support.

## Treatment Effectiveness Analysis

The evidence shows that supportive care remains the primary intervention, with specific antiviral and anti-inflammatory treatments reserved for severe cases.""",

        # Research Step 2: Drug Safety and Efficacy
        """## Drug Safety and Efficacy Analysis

My search for drug safety data in pediatric COVID-19 treatment yielded important clinical evidence:

Search results from Tavily:
[
    {
        "type": "page",
        "title": "Safety of Remdesivir in Pediatric Patients: Multi-center Study",
        "url": "https://www.nejm.org/doi/full/10.1056/NEJMc2033352",
        "content": "A multi-center observational study evaluating the safety and efficacy of remdesivir in 53 pediatric patients with severe COVID-19. The study found generally favorable safety profiles.",
        "score": 0.88,
        "raw_content": "BACKGROUND: Limited data exist on the use of remdesivir in pediatric patients with COVID-19..."
    },
    {
        "type": "page",
        "title": "Dexamethasone Use in Pediatric COVID-19: Evidence Review",
        "url": "https://jamanetwork.com/journals/jamapediatrics/fullarticle/2773698",
        "content": "Comprehensive review of dexamethasone use in pediatric COVID-19 patients, examining both benefits and potential adverse effects.",
        "score": 0.91
    },
    {
        "type": "page",
        "title": "WHO Guidelines: COVID-19 Treatment in Children",
        "url": "https://www.who.int/publications/i/item/WHO-2019-nCoV-therapeutics-2021.3",
        "content": "World Health Organization guidelines for the clinical management of COVID-19 in pediatric populations, updated with latest evidence.",
        "score": 0.87
    }
]

The evidence suggests that while adult treatments show promise, pediatric-specific safety data is still limited and requires careful monitoring.""",

        # Research Step 3: Observational Studies  
        """## Real-World Evidence from Observational Studies

I conducted additional searches for observational data on pediatric COVID-19 treatments:

Web search tool results:
[
    {
        "type": "page",
        "title": "Real-world outcomes of COVID-19 treatments in pediatric patients",
        "url": "https://journals.lww.com/pidj/fulltext/2021/07000/real_world_outcomes.12.aspx",
        "content": "Retrospective cohort study examining outcomes in 847 pediatric patients treated for COVID-19 across 15 medical centers.",
        "score": 0.83
    },
    {
        "type": "page",
        "title": "Expert Opinion: Pediatric COVID-19 Treatment Approaches", 
        "url": "https://academic.oup.com/jpids/article/10/2/123/6123456",
        "content": "Expert consensus on treatment approaches for pediatric COVID-19, synthesizing available evidence and clinical experience.",
        "score": 0.79
    }
]

This real-world evidence complements the clinical trial data and provides insights into practical implementation of treatment guidelines."""
    ]
    
    print(f"Testing extraction from {len(simulated_researcher_outputs)} researcher step results...")
    
    # Test the extraction function
    extracted_sources = extract_tavily_sources_from_observations(simulated_researcher_outputs)
    
    print(f"\n✅ Successfully extracted {len(extracted_sources)} Tavily sources")
    
    # Display extracted sources
    print("\n📊 Extracted Sources:")
    for i, source in enumerate(extracted_sources, 1):
        print(f"  {i}. {source.get('title', 'Unknown Title')}")
        print(f"     URL: {source.get('url', 'No URL')}")
        print(f"     Score: {source.get('score', 'No Score')}")
        print()
    
    # Test the full EBM pyramid generation
    if extracted_sources:
        print("🎯 Testing full EBM pyramid generation...")
        pyramid_path = generate_enhanced_ebm_pyramid_for_research(simulated_researcher_outputs)
        
        if pyramid_path:
            print(f"✅ EBM pyramid successfully generated: {pyramid_path}")
            print("\n🔍 The pyramid should now show:")
            print("   - Evidence levels (Systematic Reviews → Expert Opinion)")
            print("   - Source quality indicators")
            print("   - Favicon logos from medical institutions")
            print("   - Domain authority analysis")
        else:
            print("❌ EBM pyramid generation failed")
    else:
        print("❌ No sources extracted - cannot generate pyramid")

if __name__ == "__main__":
    test_researcher_tavily_extraction() 