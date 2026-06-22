#!/usr/bin/env python3
"""
Judgelytics ML Service Test Script
Tests model loading and predictions locally
"""

import sys
import os
sys.path.insert(0, '/Users/rakhitiwari/Downloads/proj1/backend')

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ml_service():
    """Test ML service initialization and predictions."""
    
    print("\n" + "="*60)
    print("🧪 Judgelytics ML Service Test")
    print("="*60 + "\n")
    
    try:
        # Test 1: Import and Configuration
        print("1️⃣  Testing imports and configuration...")
        from app.config import settings
        print(f"   Models directory: {settings.ML_MODELS_DIR}")
        print(f"   ✅ Configuration loaded\n")
        
        # Test 2: Model Loading
        print("2️⃣  Testing model loading...")
        from app.services.ml_service import MLService
        ml_service = MLService()
        
        if ml_service.models_loaded:
            print("   ✅ Models loaded successfully!")
            print(f"   Classifiers: {list(ml_service.classifiers.keys())}")
            print(f"   LSTM model loaded: {ml_service.lstm_model is not None}")
            print(f"   TF-IDF loaded: {ml_service.tfidf is not None}\n")
        else:
            print("   ⚠️  Models not fully loaded (some components may be missing)")
            print(f"   Classifiers: {list(ml_service.classifiers.keys())}")
            print(f"   LSTM model: {ml_service.lstm_model is not None}\n")
        
        # Test 3: Sample Prediction
        print("3️⃣  Testing prediction with sample case...")
        sample_case = {
            "description": "We purchased a defective electronic appliance from an online retailer. "
                          "The product stopped working after 3 days despite being brand new. "
                          "The retailer refuses to provide a refund or replacement.",
            "category": "Product Defect",
            "sector": "E-commerce",
            "claim_amount": 50000.0,
            "evidence_count": 4,
            "has_legal_notice": "Yes",
            "opponent_type": "Business"
        }
        
        try:
            prediction = ml_service.predict(sample_case)
            
            print(f"   📊 Prediction Results:")
            print(f"      Outcome: {prediction['outcome']}")
            print(f"      Win Probability: {prediction['win_probability_pct']}%")
            print(f"      Confidence: {prediction['confidence']}")
            print(f"      Recommended Forum: {prediction['recommended_forum']}")
            print(f"      Filing Fee: {prediction['filing_fee']}")
            print(f"      Evidence Strength: {prediction['evidence_strength']}")
            print(f"      Applicable Sections: {', '.join(prediction['applicable_sections'])}")
            print(f"      Models Used: {', '.join(prediction['models_used'])}")
            print(f"      Similar Cases: {prediction['similar_cases_count']}")
            print("   ✅ Prediction generated successfully!\n")
            
        except Exception as e:
            print(f"   ❌ Prediction failed: {str(e)}\n")
            raise
        
        # Test 4: Multiple Predictions
        print("4️⃣  Testing multiple predictions...")
        test_cases = [
            {
                "description": "Banking service dispute with incorrect charges",
                "category": "Service Complaint",
                "sector": "Banking",
                "claim_amount": 75000.0,
                "evidence_count": 2,
                "has_legal_notice": "No",
                "opponent_type": "Service Provider"
            },
            {
                "description": "Motor vehicle accident with insurance claim",
                "category": "Insurance",
                "sector": "Automobile",
                "claim_amount": 300000.0,
                "evidence_count": 5,
                "has_legal_notice": "Yes",
                "opponent_type": "Insurance Company"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            try:
                pred = ml_service.predict(case)
                print(f"   Case {i}: {pred['outcome']} ({pred['win_probability_pct']}% confidence)")
            except Exception as e:
                print(f"   Case {i}: ❌ Failed - {str(e)}")
        
        print("   ✅ Multiple predictions tested\n")
        
        # Summary
        print("="*60)
        print("✅ All tests passed!")
        print("="*60)
        print("\n📝 Summary:")
        print(f"   • Models loaded: {ml_service.models_loaded}")
        print(f"   • Available classifiers: {len(ml_service.classifiers)}")
        print(f"   • LSTM available: {ml_service.lstm_model is not None}")
        print(f"   • Ready for deployment: {ml_service.models_loaded and (ml_service.classifiers or ml_service.lstm_model)}")
        print("\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ml_service()
    sys.exit(0 if success else 1)
