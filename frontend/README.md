# Frontend

This folder contains the React review workspace scaffold for the highlight pipeline UI.

## Stack

- Vite
- React
- TypeScript
- Zustand

## Start

```powershell
cd frontend
npm install
npm run dev
```

To use the local Python backend instead of the in-memory mock layer:

```powershell
python dev_server.py --host 127.0.0.1 --port 18081
$env:VITE_API_BASE_URL="http://127.0.0.1:18081"
cd frontend
npm run dev -- --host 0.0.0.0 --port 18080
```

## Current Scope

- mock review workspace bootstrapped from local data
- componentized review layout
- Zustand store for clip selection and editing state
- lightweight Python review API available through `dev_server.py`
