# Store assets sources

Regenerate from the project root:

```bash
python3 store_assets/_source/icon.py . store_assets      # app icons (iOS, Android, stores)
python3 store_assets/_source/compose.py store_assets     # screenshots + Play feature graphic
```

- `raw_*` — device captures (Galaxy S21 Ultra; `_tab` = screen set to 2064x2752 for the iPad layout).
- `real_clean.png` / `real_final_final.png` — the worksheet before / after the app's erase (hero image).
- Captions live in `COPY` in `compose.py`.
