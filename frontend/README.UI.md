# UI Theme Notes

- Palette: slate neutrals with indigo→cyan accent (`app/globals.css`).
- Components: see `components/brand` and `components/ui`.
- APIs: uses `NEXT_PUBLIC_BACKEND_URL` for `/api/search`, `/api/analyze`, `/api/report/pdf/:id`.
- Layout: `app/page.tsx` with tabs for Overview, Financials, Transcript, Peers, Graph, Data.
- Animations respect `prefers-reduced-motion`.
- Tweak tokens in `tailwind.config.ts` and CSS variables in `app/globals.css`.
