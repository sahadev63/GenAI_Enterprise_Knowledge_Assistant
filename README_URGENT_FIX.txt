URGENT FIX - GenAI Enterprise Knowledge Assistant

Root cause fixed:
The app kept a module-level Chroma Collection object. After Reset Knowledge Base
deleted/recreated the Chroma database, Streamlit continued using the old
Collection object, producing:
chromadb.errors.NotFoundError: Collection <id> does not exist.

Fix:
Use get_collection() on every operation so the current collection is resolved
after a reset.

1. Replace:
   app/retrieval/vector_store.py
   with the vector_store.py in this ZIP.

2. Stop Streamlit completely:
   Ctrl+C

3. From the project root run:
   rmdir /s /q data\vectorstore
   mkdir data\vectorstore

4. Start again:
   streamlit run app.py

5. Upload test_policy.pdf again.

6. Test:
   How many annual leave days are employees eligible for?

Expected:
   Employees are entitled to 24 days of annual leave every year.

Then test:
   What are the annual leave and maternity leave policies?

Expected:
   Annual leave: Employees are entitled to 24 days of annual leave every year.
   Maternity leave: Information about maternity leave policies was not found
   in the provided documents.

IMPORTANT:
Do not copy the old data/vectorstore folder back into the project. It is a
runtime database and must be rebuilt from the uploaded documents.
