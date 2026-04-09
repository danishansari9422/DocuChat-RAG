"""
Test script to verify Gemini LLM integration
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gemini_integration():
    """Test Gemini LLM integration."""
    print("Testing Gemini LLM Integration")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please add your Gemini API key to .env file")
        return False
    
    print("✅ GEMINI_API_KEY found")
    
    try:
        # Test imports
        from rag.llm import GeminiLLM
        print("✅ GeminiLLM import successful")
        
        # Initialize LLM
        llm = GeminiLLM("gemini-2.5-flash")
        print("✅ Gemini LLM initialized successfully")
        
        # Test model info
        model_info = llm.get_model_info()
        print(f"✅ Model: {model_info['model_name']}")
        print(f"✅ Provider: {model_info['provider']}")
        
        # Test simple generation
        print("\n🔄 Testing simple generation...")
        test_context = """
Page 1:
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

Page 2:
Common machine learning algorithms include linear regression, decision trees, and neural networks.
"""
        
        test_question = "What are some common machine learning algorithms?"
        
        result = llm.generate(test_context, test_question)
        
        if result['success']:
            print("✅ Generation successful!")
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
            return True
        else:
            print(f"❌ Generation failed: {result['answer']}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install missing dependencies: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_gemini_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Gemini integration test passed!")
        print("Your DocuChat-RAG app is ready to use with Gemini!")
    else:
        print("❌ Test failed. Please check the error messages above.")
        print("💡 Make sure to:")
        print("   1. Add your Gemini API key to .env file")
        print("   2. Install google-generativeai: pip install google-generativeai")
        print("   3. Check your internet connection")
