1) Put inversion_backend.py and Dilated_ResNet_200.pth in the same folder.
2) Start the backend:
   python inversion_backend.py
3) Start the frontend in your Vite project:
   npm run dev
4) Open http://localhost:5173 and drag in a CSV spectrum file.

Notes:
- The backend expects a two-column CSV.
- It auto-detects whether the first column is already in THz or is omega scaled by 1e14, following your notebook convention.
- The model input uses the normalized 50-230 THz spectrum directly, matching the training dataset logic in the notebook.
