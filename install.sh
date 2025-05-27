#!/usr/bin/env bash
pip install -r requirements.txt
python -m playwright install
conda install -y -c conda-forge nodejs
cd tools
npm install playwright playwright-extra
npm install puppeteer-extra-plugin-stealth
npm install @mozilla/readability jsdom
npx playwright install --with-deps