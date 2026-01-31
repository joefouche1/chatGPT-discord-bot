"""
Test script for Phase 1 Memory System
Verifies basic functionality of HybridMemoryStorage
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from memory import (
    HybridMemoryStorage,
    MemoryEntry,
    MemoryType,
    MemoryConfidence,
    OpenAIEmbeddingProvider
)


async def test_basic_functionality():
    """Test basic memory storage operations"""
    print("🧪 Testing Phase 1 Memory System\n")
    
    # Create test memory directory
    test_memory_path = Path("test_memory")
    
    try:
        # Initialize storage with OpenAI embeddings
        print("1️⃣ Initializing HybridMemoryStorage...")
        storage = HybridMemoryStorage(base_path=test_memory_path)
        print(f"   ✓ Storage initialized at {test_memory_path}")
        print(f"   ✓ Embedding provider: {storage.embedding_provider.get_model_name()}")
        print()
        
        # Test 1: Save daily note
        print("2️⃣ Testing daily note save...")
        memory1 = MemoryEntry(
            content="User mentioned they live in Pittsburgh and work as a software engineer",
            memory_type=MemoryType.FACT,
            channel_id="test-channel-123",
            user_id="user-456",
            confidence=MemoryConfidence.HIGH,
            metadata={"source": "test"}
        )
        await storage.save_daily_note(memory1, channel_name="test-channel", guild_name="test-guild")
        
        # Check if daily log was created
        today = datetime.now().strftime("%Y-%m-%d")
        daily_log = test_memory_path / "daily" / f"{today}.md"
        assert daily_log.exists(), "Daily log file was not created"
        print(f"   ✓ Daily note saved to {daily_log}")
        print()
        
        # Test 2: Add long-term memory
        print("3️⃣ Testing long-term memory save...")
        memory2 = MemoryEntry(
            content="User prefers concise technical explanations without too much verbosity",
            memory_type=MemoryType.PREFERENCE,
            channel_id="test-channel-123",
            user_id="user-456",
            confidence=MemoryConfidence.HIGH
        )
        await storage.add_long_term_memory(memory2)
        
        # Check if MEMORY.md was created
        memory_md = test_memory_path / "long-term" / "MEMORY.md"
        assert memory_md.exists(), "MEMORY.md was not created"
        print(f"   ✓ Long-term memory saved to {memory_md}")
        
        # Check if vector DB was initialized
        vector_db = test_memory_path / "search" / "embeddings.sqlite"
        assert vector_db.exists(), "Vector database was not created"
        print(f"   ✓ Vector embedding indexed in {vector_db}")
        print()
        
        # Test 3: Add more memories for search testing
        print("4️⃣ Adding more memories for search testing...")
        memory3 = MemoryEntry(
            content="User is interested in machine learning and AI technologies",
            memory_type=MemoryType.PREFERENCE,
            channel_id="test-channel-123",
            user_id="user-456"
        )
        await storage.add_long_term_memory(memory3)
        
        memory4 = MemoryEntry(
            content="Discussed the Steelers game last Sunday",
            memory_type=MemoryType.EVENT,
            channel_id="test-channel-123"
        )
        await storage.add_long_term_memory(memory4)
        print("   ✓ Added 2 additional memories")
        print()
        
        # Test 4: Search memories
        print("5️⃣ Testing semantic search...")
        
        # Search for location
        print("   Query 1: 'Where does the user live?'")
        results = await storage.search_memories("Where does the user live?", top_k=3)
        print(f"   → Found {len(results)} results")
        if results:
            print(f"   → Top result: {results[0].memory.content[:60]}... (score: {results[0].score:.3f})")
        print()
        
        # Search for preferences
        print("   Query 2: 'What are the user's preferences?'")
        results = await storage.search_memories("What are the user's preferences?", top_k=3)
        print(f"   → Found {len(results)} results")
        if results:
            print(f"   → Top result: {results[0].memory.content[:60]}... (score: {results[0].score:.3f})")
        print()
        
        # Search for sports
        print("   Query 3: 'Tell me about sports'")
        results = await storage.search_memories("Tell me about sports", top_k=3)
        print(f"   → Found {len(results)} results")
        if results:
            print(f"   → Top result: {results[0].memory.content[:60]}... (score: {results[0].score:.3f})")
        print()
        
        # Test 5: Filtered search
        print("6️⃣ Testing filtered search (only PREFERENCE type)...")
        results = await storage.search_memories(
            "user preferences",
            top_k=5,
            memory_type=MemoryType.PREFERENCE
        )
        print(f"   → Found {len(results)} PREFERENCE results")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.memory.content[:50]}... (score: {result.score:.3f})")
        print()
        
        # Test 6: Read recent logs
        print("7️⃣ Testing recent daily logs retrieval...")
        recent_logs = await storage.get_recent_daily_logs(days=7)
        print(f"   ✓ Retrieved {len(recent_logs)} recent daily logs")
        print()
        
        # Test 7: Read long-term memory
        print("8️⃣ Testing long-term memory retrieval...")
        memory_content = await storage.get_long_term_memory()
        print(f"   ✓ Retrieved MEMORY.md ({len(memory_content)} characters)")
        print()
        
        print("✅ All tests passed!\n")
        print("📂 Test files created in:", test_memory_path.absolute())
        print("   - Daily logs:", (test_memory_path / "daily").absolute())
        print("   - Long-term memory:", (test_memory_path / "long-term").absolute())
        print("   - Vector database:", (test_memory_path / "search").absolute())
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def main():
    success = await test_basic_functionality()
    
    if success:
        print("\n" + "="*60)
        print("Phase 1 implementation is working correctly! 🎉")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Phase 1 implementation has issues ⚠️")
        print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
