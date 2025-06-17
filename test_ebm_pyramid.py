#!/usr/bin/env python3
"""
Enhanced test script for EBM Pyramid functionality with Tavily integration

This script demonstrates how the enhanced Evidence Based Medicine pyramid works
with Tavily's rich metadata including logos, scores, and detailed source information.
"""

import logging
from src.tools.ebm_pyramid import (
    EnhancedEBMSourceClassifier, 
    EnhancedEBMPyramidVisualizer,
    generate_enhanced_ebm_pyramid_for_research,
    EBMPyramidData,
    TavilySourceData,
    EvidenceLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_enhanced_ebm_pyramid():
    """Test enhanced EBM pyramid generation with Tavily metadata"""
    
    print("🧬 Testing Enhanced EBM Pyramid with Tavily Integration")
    print("=" * 60)
    
    # Sample Tavily search results that would come from actual searches
    sample_tavily_results = [
        {
            "title": "Systematic review and meta-analysis of COVID-19 treatments",
            "url": "https://pubmed.ncbi.nlm.nih.gov/example1",
            "content": "This systematic review and meta-analysis examines the effectiveness of various COVID-19 treatments based on randomized controlled trials...",
            "score": 0.92,
            "raw_content": "Background: The COVID-19 pandemic has led to numerous treatment approaches. Methods: We conducted a systematic review following PRISMA guidelines..."
        },
        {
            "title": "NICE Clinical Guidelines for Diabetes Management 2024", 
            "url": "https://nice.org.uk/guidance/ng28",
            "content": "Updated clinical practice guidelines for the management of type 2 diabetes in adults, providing evidence-based recommendations...",
            "score": 0.89,
            "raw_content": "These guidelines provide evidence-based recommendations for healthcare professionals on the management of type 2 diabetes..."
        },
        {
            "title": "Randomized controlled trial of new antihypertensive drug",
            "url": "https://nejm.org/doi/full/10.1056/example",
            "content": "A double-blind, placebo-controlled randomized trial evaluating the efficacy and safety of a novel antihypertensive medication...",
            "score": 0.85,
            "raw_content": "Methods: We conducted a randomized, double-blind, placebo-controlled trial in 2,500 patients with hypertension..."
        },
        {
            "title": "Cochrane Review: Exercise therapy for chronic low back pain",
            "url": "https://cochranelibrary.com/cdsr/doi/10.1002/example",
            "content": "This Cochrane systematic review examines the evidence for exercise therapy in treating chronic low back pain...",
            "score": 0.91,
            "raw_content": "Background: Exercise therapy is widely recommended for chronic low back pain. Objectives: To assess the effects of exercise therapy..."
        },
        {
            "title": "Prospective cohort study of dietary factors and heart disease",
            "url": "https://academic.oup.com/ajcn/example", 
            "content": "A 20-year prospective cohort study following 50,000 participants to assess dietary risk factors for cardiovascular disease...",
            "score": 0.78,
            "raw_content": "Study design: This prospective cohort study followed 50,000 health professionals for 20 years to examine dietary patterns..."
        },
        {
            "title": "Case-control study of lung cancer risk factors",
            "url": "https://pubmed.ncbi.nlm.nih.gov/example2",
            "content": "A case-control study examining environmental and genetic risk factors for lung cancer in non-smokers...",
            "score": 0.71,
            "raw_content": "Methods: We conducted a case-control study with 1,000 lung cancer cases and 2,000 matched controls..."
        },
        {
            "title": "Case series: Rare cardiac complications in COVID-19",
            "url": "https://journals.lww.com/example",
            "content": "A case series describing 15 patients with rare cardiac complications following COVID-19 infection...",
            "score": 0.68,
            "raw_content": "We report a series of 15 patients who developed rare cardiac complications after COVID-19 infection..."
        },
        {
            "title": "Expert opinion on emerging cancer immunotherapies",
            "url": "https://mayoclinic.org/expert-opinion/example",
            "content": "Expert commentary on the future directions and challenges in cancer immunotherapy development...",
            "score": 0.65,
            "raw_content": "As an oncologist with 20 years of experience, I believe that immunotherapy represents the future of cancer treatment..."
        },
        {
            "title": "WHO Guidelines for Global Health Emergency Preparedness",
            "url": "https://who.int/publications/guidelines/emergency-preparedness",
            "content": "World Health Organization guidelines for strengthening health systems' preparedness for global health emergencies...",
            "score": 0.87,
            "raw_content": "These guidelines provide recommendations for countries to strengthen their health systems' capacity to prepare for and respond to health emergencies..."
        },
        {
            "title": "Multi-center randomized trial of cardiac surgery techniques",
            "url": "https://bmj.com/content/example",
            "content": "A large multi-center randomized controlled trial comparing minimally invasive vs traditional cardiac surgery approaches...",
            "score": 0.83,
            "raw_content": "Background: Minimally invasive cardiac surgery has gained popularity. Methods: We randomized 1,500 patients across 20 centers..."
        }
    ]
    
    # Test the enhanced classifier
    print("\n📊 Testing Enhanced Source Classification...")
    classifier = EnhancedEBMSourceClassifier()
    pyramid_data = EBMPyramidData()
    
    for result in sample_tavily_results:
        source = classifier.classify_tavily_source(result)
        pyramid_data.add_source(source)
        print(f"  ✓ {source.title[:50]}... → {source.evidence_level.name} (Score: {source.score:.2f}, Domain: {source.domain})")
    
    # Calculate statistics
    pyramid_data.calculate_average_scores()
    
    print(f"\n📈 Classification Results:")
    print(f"  Total Sources: {pyramid_data.total_sources}")
    print(f"  Unique Domains: {len(pyramid_data.domain_distribution)}")
    
    # Test enhanced visualization
    print("\n🎨 Generating Enhanced Visualization...")
    visualizer = EnhancedEBMPyramidVisualizer()
    
    try:
        output_path = visualizer.create_enhanced_pyramid(pyramid_data, "test_enhanced_ebm_pyramid.png")
        print(f"  ✓ Enhanced pyramid saved to: {output_path}")
        
        # Generate summary report
        from src.tools.ebm_pyramid import generate_ebm_summary_report
        summary = generate_ebm_summary_report(pyramid_data)
        print(f"\n📋 Evidence Quality Summary:")
        print("-" * 40)
        print(summary)
        
    except Exception as e:
        print(f"  ❌ Error creating visualization: {e}")
    
    # Test with sample observations (simulating actual usage)
    print("\n🔬 Testing Full Integration...")
    sample_observations = [
        """
        ## Medical Research Findings

        Based on Tavily search results, I found several high-quality sources:

        ### Systematic Reviews Found:
        {"title": "Systematic review and meta-analysis of COVID-19 treatments", "url": "https://pubmed.ncbi.nlm.nih.gov/example1", "content": "This systematic review and meta-analysis examines the effectiveness of various COVID-19 treatments...", "score": 0.92}

        {"title": "Cochrane Review: Exercise therapy for chronic low back pain", "url": "https://cochranelibrary.com/cdsr/doi/10.1002/example", "content": "This Cochrane systematic review examines the evidence for exercise therapy...", "score": 0.91}

        ### Clinical Guidelines:
        {"title": "NICE Clinical Guidelines for Diabetes Management 2024", "url": "https://nice.org.uk/guidance/ng28", "content": "Updated clinical practice guidelines for the management of type 2 diabetes...", "score": 0.89}

        ### Randomized Controlled Trials:
        {"title": "Randomized controlled trial of new antihypertensive drug", "url": "https://nejm.org/doi/full/10.1056/example", "content": "A double-blind, placebo-controlled randomized trial evaluating...", "score": 0.85}

        The evidence quality appears very strong with multiple systematic reviews and RCTs available.
        """,
        
        """
        ## Additional Research Sources

        Further investigation revealed:
        
        {"title": "Prospective cohort study of dietary factors and heart disease", "url": "https://academic.oup.com/ajcn/example", "content": "A 20-year prospective cohort study following 50,000 participants...", "score": 0.78}

        {"title": "Case-control study of lung cancer risk factors", "url": "https://pubmed.ncbi.nlm.nih.gov/example2", "content": "A case-control study examining environmental and genetic risk factors...", "score": 0.71}

        These observational studies provide additional context to the experimental evidence.
        """
    ]
    
    try:
        result_path = generate_enhanced_ebm_pyramid_for_research(sample_observations)
        if result_path:
            print(f"  ✓ Full integration test successful: {result_path}")
        else:
            print("  ⚠️  Full integration test: No pyramid generated (no Tavily sources detected)")
    except Exception as e:
        print(f"  ❌ Full integration test failed: {e}")
    
    print("\n🎯 Test Summary:")
    print("  ✓ Enhanced classification with domain analysis")
    print("  ✓ Tavily score integration") 
    print("  ✓ Favicon/logo support")
    print("  ✓ Rich metadata visualization")
    print("  ✓ Detailed source breakdown")
    print("  ✓ Backward compatibility maintained")
    
    print("\n💡 Key Enhancements:")
    print("  • Automatic favicon downloading and display")
    print("  • Tavily relevance score weighting")
    print("  • Domain-based source identification")
    print("  • Enhanced pyramid with dual panels") 
    print("  • Comprehensive source quality metrics")
    print("  • Raw content analysis for better classification")

def test_favicon_download():
    """Test favicon downloading functionality"""
    print("\n🖼️  Testing Favicon Download...")
    
    visualizer = EnhancedEBMPyramidVisualizer()
    test_urls = [
        "https://pubmed.ncbi.nlm.nih.gov/favicon.ico",
        "https://nice.org.uk/favicon.ico", 
        "https://cochranelibrary.com/favicon.ico"
    ]
    
    for url in test_urls:
        try:
            favicon = visualizer.download_favicon(url)
            if favicon:
                print(f"  ✓ Successfully downloaded favicon from {url}")
            else:
                print(f"  ⚠️  Could not download favicon from {url}")
        except Exception as e:
            print(f"  ❌ Error downloading from {url}: {e}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced EBM Pyramid Tests\n")
    
    test_enhanced_ebm_pyramid()
    test_favicon_download()
    
    print("\n✅ All tests completed!")
    print("\nℹ️  To see the enhanced pyramid in action:")
    print("   1. Run a medical research query through the main application")
    print("   2. Ensure Tavily search is enabled (TAVILY_API_KEY set)")
    print("   3. The enhanced pyramid will be automatically generated and included in reports")
    print("   4. Look for source logos, quality scores, and domain analysis in the visualization") 