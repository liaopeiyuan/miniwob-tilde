#!/usr/bin/env python3
"""Deploy and test the MiniWoB Modal backend."""

import argparse
import sys
from pathlib import Path
from typing import cast

def deploy_modal_app():
    """Deploy the Modal app to the cloud."""
    try:
        import modal
        print("✓ Modal is installed")
    except ImportError:
        print("✗ Modal is not installed. Please run: pip install modal")
        return False
    
    try:
        # Import the app to trigger deployment
        from miniwob.modal_instance import app
        print("✓ Modal app imported successfully")
        
        # Check if we can access Modal (authentication)
        print("📡 Checking Modal authentication...")
        
        # This will trigger the image build if needed
        print("🔨 Building Chrome image (this may take a few minutes on first run)...")
        # A short-lived run will force the image to build.
        with app.run():
            pass
        
        return True
        
    except Exception as e:
        print(f"✗ Error deploying Modal app: {e}")
        return False

def test_modal_backend():
    """Test the Modal backend with a simple task using gymnasium.make."""
    try:
        import gymnasium as gym
        # Import miniwob to ensure environments are registered.
        import miniwob
        from miniwob.action import ActionTypes
        from miniwob.environment import MiniWoBEnvironment

        print("🧪 Testing Modal backend via gymnasium.make...")
        
        miniwob_env = gym.make("miniwob/click-test-v1", backend="modal", wait_ms=100)

        print("🔄 Resetting environment...")
        obs, info = miniwob_env.reset(seed=42)
        
        if obs:
            print(f"✓ Got initial observation: {obs['utterance'][:50]}...")
        else:
            print("✗ Failed to get initial observation")
            miniwob_env.close()
            return False
        
        print("🎯 Taking test action...")
        action = cast(MiniWoBEnvironment, miniwob_env.unwrapped).create_action(
            ActionTypes.CLICK_COORDS, coords=[80, 105]
        )
        
        obs, reward, terminated, truncated, info = miniwob_env.step(action)
        
        print(f"✓ Action executed. Reward: {reward}, Terminated: {terminated}")
        
        # Clean up
        miniwob_env.close()
        print("✓ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Modal backend: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        return False
    print(f"✓ Python {sys.version}")
    
    # Check Modal installation
    try:
        import modal
        print(f"✓ Modal {modal.__version__} installed")
    except ImportError:
        print("✗ Modal not installed. Run: pip install modal")
        return False
    
    # Check if Modal is authenticated
    try:
        # This will raise an exception if not authenticated
        modal.App("test-auth-check")
        print("✓ Modal authentication configured")
    except Exception as e:
        print("✗ Modal authentication not configured. Run: modal token new")
        return False
    
    # Check MiniWoB dependencies
    try:
        from miniwob.action import ActionSpaceConfig
        from miniwob.constants import TASK_WIDTH
        print("✓ MiniWoB dependencies available")
    except ImportError as e:
        print(f"✗ MiniWoB dependencies missing: {e}")
        return False
    
    return True

def main():
    """Main deployment and testing function."""
    parser = argparse.ArgumentParser(
        description="Deploy and test MiniWoB Modal backend"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the backend test after deployment"
    )
    parser.add_argument(
        "--test-only",
        action="store_true", 
        help="Only run the test, skip deployment"
    )
    
    args = parser.parse_args()
    
    print("🚀 MiniWoB Modal Backend Deployment")
    print("=" * 40)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return 1
    
    print("\n✅ All prerequisites met!")
    
    # Deploy app
    if not args.test_only:
        print("\n📦 Deploying Modal app...")
        if not deploy_modal_app():
            print("\n❌ Deployment failed!")
            return 1
        print("\n✅ Deployment successful!")
    
    # Test backend
    if not args.skip_test:
        print("\n🧪 Testing backend...")
        if not test_modal_backend():
            print("\n❌ Backend test failed!")
            return 1
        print("\n✅ Backend test passed!")
    
    print("\n🎉 All done! Your Modal backend is ready to use.")
    print("\nNext steps:")
    print("1. Run: python example_modal_usage.py")
    print("2. Check the README_modal_backend.md for more examples")
    print("3. Monitor your usage at: https://modal.com")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
