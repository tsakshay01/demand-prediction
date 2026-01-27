"""
Verification Test: Multimodality Proof
Demonstrates that changing the product description affects the prediction.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightweight_model import LightweightMultimodalModel


def test_multimodality():
    """
    Test 1: Prove that changing description changes prediction
    (Same sales_history, different descriptions)
    """
    print("=" * 60)
    print("TEST 1: Multimodality Proof")
    print("Keeping sales_history CONSTANT, changing description")
    print("=" * 60)
    
    model = LightweightMultimodalModel()
    
    # Train the model first if not trained
    if not model.is_trained:
        print("\nModel not trained. Training now...")
        model.train('ml_service/dataset.csv')
        print()
    
    # Fixed sales history
    fixed_history = [100, 120, 110, 130, 125, 140, 135, 150, 145, 160]
    
    # Different descriptions
    descriptions = [
        "Vintage Red Cotton T-Shirt",
        "Heavy Winter Wool Jacket",
        "Lightweight Summer Beach Dress",
        "Professional Business Suit"
    ]
    
    predictions = []
    for desc in descriptions:
        result = model.predict_single(desc, fixed_history)
        pred = result.get('prediction', 0)
        predictions.append(pred)
        print(f"Description: '{desc}'")
        print(f"  -> Prediction: {pred:.2f}")
        print(f"  -> Modalities: {result.get('modalities_used', [])}")
        print()
    
    # Check if predictions are different
    unique_predictions = len(set([round(p, 2) for p in predictions]))
    
    if unique_predictions > 1:
        print("✅ PASS: Different descriptions produce different predictions!")
        print(f"   {unique_predictions} unique predictions from {len(descriptions)} descriptions")
    else:
        print("❌ FAIL: All predictions are the same (text not affecting output)")
    
    return unique_predictions > 1


def test_cold_start():
    """
    Test 2: Cold-start handling
    (No sales_history, only description)
    """
    print("\n" + "=" * 60)
    print("TEST 2: Cold-Start Handling")
    print("No sales_history, only description provided")
    print("=" * 60)
    
    model = LightweightMultimodalModel()
    
    result = model.predict_single(
        description="Brand New Product Launch - Premium Quality",
        sales_history=[]
    )
    
    print(f"Result: {result}")
    
    if 'prediction' in result and result.get('text_features_active'):
        print("✅ PASS: Cold-start prediction generated using text features")
        return True
    else:
        print("❌ FAIL: Cold-start not handled correctly")
        return False


def test_insufficient_data():
    """
    Test 3: Insufficient data handling
    (No description AND no sales_history)
    """
    print("\n" + "=" * 60)
    print("TEST 3: Insufficient Data Handling")
    print("Both description and sales_history are empty")
    print("=" * 60)
    
    model = LightweightMultimodalModel()
    
    result = model.predict_single(
        description="",
        sales_history=[]
    )
    
    print(f"Result: {result}")
    
    if 'error' in result and result['error'] == 'insufficient_data':
        print("✅ PASS: Correctly returned insufficient_data error")
        return True
    else:
        print("❌ FAIL: Should have returned error for missing data")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("LIGHTWEIGHT MULTIMODAL MODEL - VERIFICATION SUITE")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Multimodality Proof", test_multimodality()))
    results.append(("Cold-Start Handling", test_cold_start()))
    results.append(("Insufficient Data", test_insufficient_data()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️ SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
