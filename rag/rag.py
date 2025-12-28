"""
RAG (Retrieval-Augmented Generation) Implementation
This script demonstrates how to build a RAG system using LangChain, OpenAI, and FAISS.
Supports both PDF and TXT files from a documents folder.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def setup_environment():
    """Load environment variables and verify API key."""
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY in your .env file")
    
    print("✓ Environment setup complete")


def create_documents_folder():
    """Create documents folder if it doesn't exist."""
    docs_path = Path("documents")
    docs_path.mkdir(exist_ok=True)
    print(f"✓ Documents folder ready at: {docs_path.absolute()}")
    return docs_path


def load_documents_from_folder(folder_path="documents"):
    """Load all PDF and TXT documents from folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Documents folder not found: {folder_path}")
    
    all_documents = []
    
    # Load PDF files
    pdf_files = list(folder.glob("*.pdf"))
    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        all_documents.extend(documents)
        print(f"  ✓ Loaded {len(documents)} pages")
    
    # Load TXT files
    txt_files = list(folder.glob("*.txt"))
    for txt_file in txt_files:
        print(f"Loading TXT: {txt_file.name}")
        loader = TextLoader(str(txt_file), encoding='utf-8')
        documents = loader.load()
        all_documents.extend(documents)
        print(f"  ✓ Loaded {len(documents)} document(s)")
    
    if not all_documents:
        raise ValueError(f"No PDF or TXT files found in {folder_path}")
    
    print(f"\n✓ Total documents loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks, embeddings_model="text-embedding-3-small"):
    """Create FAISS vector store from document chunks."""
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✓ Vector store created")
    return vector_store


def format_docs(docs):
    """Format documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vector_store, model_name="gpt-4o-mini", temperature=0):
    """Create RAG chain with retrieval and generation using LCEL."""
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Custom prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Create RAG chain using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✓ RAG chain created")
    return rag_chain, retriever


def query_rag_system(rag_chain, retriever, question):
    """Query the RAG system and return answer with sources."""
    # Get answer
    answer = rag_chain.invoke(question)
    
    # Get source documents using invoke method
    source_docs = retriever.invoke(question)
    
    print(f"\n💡 Question: {question}")
    print(f"\n✅ Answer: {answer}")
    print(f"\n📚 Sources ({len(source_docs)} documents):")
    
    for i, doc in enumerate(source_docs, 1):
        print(f"\n--- Source {i} ---")
        source_file = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        print(f"📄 File: {Path(source_file).name}")
        print(f"📖 Page: {page}")
        print(f"📝 Content preview: {doc.page_content[:200]}...")
    
    return {"result": answer, "source_documents": source_docs}


def save_vector_store(vector_store, save_path="faiss_index"):
    """Save FAISS vector store to disk."""
    vector_store.save_local(save_path)
    print(f"✓ Vector store saved to {save_path}")


def load_vector_store(load_path="faiss_index", embeddings_model="text-embedding-3-small"):
    """Load FAISS vector store from disk."""
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    vector_store = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"✓ Vector store loaded from {load_path}")
    return vector_store


def create_sample_documents():
    """Create sample documents for testing."""
    docs_path = Path("documents")
    docs_path.mkdir(exist_ok=True)
    
    # Create sample text file 1
    sample1 = docs_path / "sample1.txt"
    sample1.write_text("""
# Introduction to Machine Learning

Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Key Concepts:
- Supervised Learning: Learning from labeled data
- Unsupervised Learning: Finding patterns in unlabeled data
- Reinforcement Learning: Learning through trial and error

Machine learning algorithms build mathematical models based on sample data, known as training data, to make predictions or decisions.
""")
    
    # Create sample text file 2
    sample2 = docs_path / "sample2.txt"
    sample2.write_text("""
# Deep Learning Overview

Deep Learning is a subset of machine learning that uses neural networks with multiple layers.

## Applications:
- Computer Vision: Image recognition and object detection
- Natural Language Processing: Text analysis and generation
- Speech Recognition: Converting speech to text

Deep learning has revolutionized AI by achieving human-level performance in many tasks.
""")
    
    # Create sample text file 3
    sample3 = docs_path / "sample3.txt"
    sample3.write_text("""
# RAG Systems Explained

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.

## How RAG Works:
1. Document Ingestion: Load and process documents
2. Embedding Creation: Convert text to vector representations
3. Vector Storage: Store embeddings in a vector database
4. Retrieval: Find relevant documents for a query
5. Generation: Use LLM to generate answers based on retrieved context

RAG systems improve accuracy by grounding LLM responses in actual documents.
""")
    
    # Create sample text file 4
    sample4 = docs_path / "sample4.txt"
    sample4.write_text("""
# Artificial Intelligence (AI)

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.

## Types of AI:
- Narrow AI: Designed for specific tasks (e.g., voice assistants, recommendation systems)
- General AI: Hypothetical AI with human-level intelligence across all domains
- Super AI: Theoretical AI that surpasses human intelligence

AI encompasses various technologies including machine learning, deep learning, natural language processing, computer vision, and robotics.
""")
    
    print(f"✓ Created 4 sample documents in {docs_path.absolute()}")
    print("  - sample1.txt (Machine Learning)")
    print("  - sample2.txt (Deep Learning)")
    print("  - sample3.txt (RAG Systems)")
    print("  - sample4.txt (Artificial Intelligence)")


def interactive_mode(rag_chain, retriever):
    """Run interactive Q&A session."""
    print("\n" + "="*80)
    print("🤖 Interactive Q&A Mode")
    print("="*80)
    print("Type your questions below. Type 'exit', 'quit', or 'q' to stop.")
    print("="*80 + "\n")
    
    while True:
        try:
            question = input("\n🤔 Your question: ").strip()
            
            if not question:
                print("⚠️  Please enter a question.")
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            print("\n🔍 Searching documents and generating answer...\n")
            query_rag_system(rag_chain, retriever, question)
            print("\n" + "-"*80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


def main():
    """Main function to run the RAG pipeline."""
    # Setup
    setup_environment()
    
    # Create documents folder
    docs_folder = create_documents_folder()
    
    # Check if documents exist, if not create samples
    existing_files = list(docs_folder.glob("*.pdf")) + list(docs_folder.glob("*.txt"))
    if not existing_files:
        print("\nNo documents found. Creating sample documents...")
        create_sample_documents()
        print("\nYou can add your own PDF or TXT files to the 'documents' folder.\n")
    
    # Load and process documents
    documents = load_documents_from_folder("documents")
    chunks = split_documents(documents)
    
    # Create vector store
    vector_store = create_vector_store(chunks)
    
    # Optional: Save vector store for later use
    # save_vector_store(vector_store)
    
    # Create RAG chain
    rag_chain, retriever = create_rag_chain(vector_store)
    
    # Check if running in interactive mode or with command line questions
    if len(sys.argv) > 1:
        # Command line mode - answer the question provided as argument
        question = " ".join(sys.argv[1:])
        print("\n" + "="*80)
        print("📝 Single Question Mode")
        print("="*80 + "\n")
        query_rag_system(rag_chain, retriever, question)
    else:
        # Interactive mode - continuous Q&A
        interactive_mode(rag_chain, retriever)


if __name__ == "__main__":
    main()