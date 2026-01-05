"""
Production Inference Script
===========================
Standalone script for running predictions on the trained model.

Usage:
    # Single prediction
    python inference.py --single '{"Air temperature [K]": 298.1, ...}'
    
    # Batch prediction from file
    python inference.py --batch data/processed/test_data.csv
    
    # Interactive mode
    python inference.py --interactive

Author: IoT Predictive Maintenance Team
"""
import argparse
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.api.inference import InferenceEngine, predict_single, predict_batch
from src.config import MODELS_DIR, get_processed_data_path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Predictive Maintenance Inference Tool'
    )
    
    parser.add_argument(
        '--single', '-s',
        type=str,
        help='JSON string for single prediction'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Path to CSV file for batch prediction'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--no-explanation',
        action='store_true',
        help='Disable SHAP explanations'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=str(MODELS_DIR),
        help='Path to model directory'
    )
    
    return parser.parse_args()


def run_single_prediction(engine: InferenceEngine, json_str: str, include_explanation: bool = True):
    """Run single prediction from JSON string."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return
    
    result = engine.predict(data, include_explanation=include_explanation)
    print(json.dumps(result, indent=2, default=str))


def run_batch_prediction(engine: InferenceEngine, file_path: str):
    """Run batch prediction from CSV file."""
    import pandas as pd
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    records = df.to_dict(orient='records')
    
    print(f"Processing {len(records)} records...")
    result = engine.predict_batch(records, include_explanation=False)
    
    print(f"\nResults Summary:")
    print(f"  Total records: {result['total_records']}")
    print(f"  Successful: {result['successful']}")
    print(f"  Failed: {result['failed']}")
    print(f"  Inference time: {result['metadata']['batch_inference_time_ms']:.2f}ms")
    print(f"  Avg time per record: {result['metadata']['avg_time_per_record_ms']:.2f}ms")
    
    # Show predictions
    print("\nPredictions:")
    for pred in result['predictions'][:10]:  # Show first 10
        risk = pred['risk_level']
        prob = pred['failure_probability']
        print(f"  Record {pred['index']}: {risk} risk ({prob:.1%} failure probability)")
    
    if len(result['predictions']) > 10:
        print(f"  ... and {len(result['predictions']) - 10} more")


def run_interactive_mode(engine: InferenceEngine):
    """Run interactive prediction mode."""
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE - Interactive Mode")
    print("=" * 60)
    print("\nEnter sensor readings (or 'quit' to exit):\n")
    
    while True:
        try:
            # Get inputs
            print("-" * 40)
            air_temp = input("Air temperature [K] (e.g., 298.1): ").strip()
            if air_temp.lower() == 'quit':
                break
            
            proc_temp = input("Process temperature [K] (e.g., 308.6): ").strip()
            rpm = input("Rotational speed [rpm] (e.g., 1551): ").strip()
            torque = input("Torque [Nm] (e.g., 42.8): ").strip()
            tool_wear = input("Tool wear [min] (e.g., 100): ").strip()
            
            # Make prediction
            data = {
                'Air temperature [K]': float(air_temp),
                'Process temperature [K]': float(proc_temp),
                'Rotational speed [rpm]': float(rpm),
                'Torque [Nm]': float(torque),
                'Tool wear [min]': float(tool_wear)
            }
            
            result = engine.predict(data, include_explanation=True)
            
            # Display result
            print("\n" + "=" * 40)
            print("PREDICTION RESULT")
            print("=" * 40)
            
            if result['success']:
                risk = result['risk_level']
                prob = result['failure_probability']
                action = result['recommended_action']
                
                # Color coding for risk level
                risk_colors = {
                    'CRITICAL': '\033[91m',  # Red
                    'HIGH': '\033[93m',      # Yellow
                    'MEDIUM': '\033[94m',    # Blue
                    'LOW': '\033[92m'        # Green
                }
                reset = '\033[0m'
                
                color = risk_colors.get(risk, '')
                print(f"Risk Level: {color}{risk}{reset}")
                print(f"Failure Probability: {prob:.1%}")
                print(f"Recommended Action: {action.replace('_', ' ').title()}")
                
                if result.get('explanation'):
                    print("\nKey Factors:")
                    for factor in result['explanation']['top_factors'][:3]:
                        direction = "↑" if factor['direction'] == 'increases_risk' else "↓"
                        print(f"  {direction} {factor['feature']}: {factor['shap_value']:.3f}")
            else:
                print("Prediction failed:")
                for error in result['errors']:
                    print(f"  - {error}")
            
            print()
            
        except ValueError as e:
            print(f"\nError: Invalid input - {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize engine
    print("Loading model...")
    engine = InferenceEngine(args.model_path)
    print("Model loaded successfully!\n")
    
    include_explanation = not args.no_explanation
    
    if args.single:
        run_single_prediction(engine, args.single, include_explanation)
    elif args.batch:
        run_batch_prediction(engine, args.batch)
    elif args.interactive:
        run_interactive_mode(engine)
    else:
        # Default: run example prediction
        print("Running example prediction...")
        print("-" * 40)
        
        sample_input = {
            'Air temperature [K]': 298.1,
            'Process temperature [K]': 308.6,
            'Rotational speed [rpm]': 1551,
            'Torque [Nm]': 42.8,
            'Tool wear [min]': 100,
            'machine_id': 'EXAMPLE_001'
        }
        
        print("Input:")
        for key, value in sample_input.items():
            print(f"  {key}: {value}")
        
        result = engine.predict(sample_input, include_explanation=True)
        
        print("\nOutput:")
        print(json.dumps(result, indent=2, default=str))
        
        print("\n" + "-" * 40)
        print("Usage Examples:")
        print("  python inference.py --interactive")
        print("  python inference.py --single '{\"Air temperature [K]\": 298.1, ...}'")
        print("  python inference.py --batch data/test.csv")


if __name__ == "__main__":
    main()
