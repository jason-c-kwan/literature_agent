const { chromium } = require('playwright-extra');
const stealth = require('puppeteer-extra-plugin-stealth')(); // Yes, puppeteer-extra-plugin-stealth works with playwright-extra
const { Readability } = require('@mozilla/readability');
const { JSDOM } = require('jsdom');

// Add stealth plugin
chromium.use(stealth);

const url = process.argv[2];

if (!url) {
    console.error(JSON.stringify({ status: 'error', message: 'No URL provided', url: null }));
    process.exit(1);
}

(async () => {
    let browser;
    const screenshotPath = 'debug_screenshot.png'; // Screenshot in project root
    try {
        browser = await chromium.launch({
            headless: false, // Run in headful mode
            // You can add proxy settings here if needed, e.g.:
            // proxy: {
            //   server: 'http://your-proxy-server:port',
            //   username: 'your-proxy-username', // optional
            //   password: 'your-proxy-password'  // optional
            // }
        });
        const context = await browser.newContext({
            userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            viewport: { width: 1920, height: 1080 }
        });
        const page = await context.newPage();

        await page.goto(url, { waitUntil: 'networkidle', timeout: 90000 }); // Increased to 90s timeout

        // Optional: longer random delay for observation in headful mode
        await page.waitForTimeout(Math.floor(Math.random() * 5000) + 5000); // 5-10s delay

        await page.screenshot({ path: screenshotPath, fullPage: true });
        // console.error(`Screenshot saved to ${screenshotPath}`); // Log to stderr for visibility

        const htmlContent = await page.content();
        const finalUrl = page.url();

        // Attempt to extract readable content
        let readableHtml = '';
        let mainContent = '';
        try {
            const doc = new JSDOM(htmlContent, { url: finalUrl });
            const reader = new Readability(doc.window.document);
            const article = reader.parse();
            if (article && article.content) {
                readableHtml = article.content; // This is HTML string of the main content
                mainContent = article.textContent; // This is plain text
            }
        } catch (readabilityError) {
            // console.error('Readability extraction failed:', readabilityError.message);
            // Fallback to full HTML if Readability fails
        }
        
        // Output JSON
        // If readableHtml is available and substantial, use it. Otherwise, consider full htmlContent.
        // For now, let's prioritize readableHtml if it exists.
        // The Python side will decide if it's good enough or if it needs to fall back further.
        const output = {
            status: 'success',
            html: readableHtml || htmlContent, // Prefer readable, fallback to full
            main_text: mainContent, // Plain text from readability
            final_url: finalUrl,
            source: 'nodejs_stealth_fetcher'
        };
        console.log(JSON.stringify(output));

    } catch (error) {
        // Ensure all errors are stringified JSON to stderr
        let errorMessage = error.message;
        if (error.stack) {
            errorMessage += '\\n' + error.stack;
        }
        console.error(JSON.stringify({ 
            status: 'error', 
            message: errorMessage, 
            url: url, 
            source: 'nodejs_stealth_fetcher' 
        }));
        process.exitCode = 1; // Indicate failure
    } finally {
        if (browser) {
            await browser.close();
        }
    }
})();
