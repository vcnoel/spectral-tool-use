import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from spectral_guardrails.probes.labeling import assign_label

def test_labeling_logic():
    print("Testing labeling logic...")
    
    # Case 1: exact match → label 0
    assert assign_label("calculate_loan_payment", "calculate_loan_payment") == 0
    
    # Case 2: case/underscore normalization → label 0
    assert assign_label("Calculate_Loan_Payment", "calculateloanpayment") == 0
    
    # Case 3: wrong function → label 1
    assert assign_label("calculate_distance", "calculate_loan_payment") == 1
    
    # Case 4: empty prediction (tool bypass) → label 1
    assert assign_label("", "calculate_loan_payment") == 1
    
    # Case 5: partial prefix match does NOT count as correct
    assert assign_label("calculate_loan", "calculate_loan_payment") == 1
    
    # Case 6: Parentheses handling
    assert assign_label("calculate_loan_payment(100)", "calculate_loan_payment") == 0

    print("Labeling logic: PASS")

if __name__ == "__main__":
    try:
        test_labeling_logic()
        print("\nALL LABEL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
