const { chromium } = require('playwright-extra');
const stealth = require('puppeteer-extra-plugin-stealth')();
const { Readability } = require('@mozilla/readability');
const { JSDOM } = require('jsdom');
const fs = require('fs'); // For checking directory
const path = require('path'); // For joining paths

chromium.use(stealth);

const url = process.argv[2];
const downloadsDir = path.join(process.cwd(), 'workspace', 'downloads', 'stealth_dl');

if (!url) {
    console.error(JSON.stringify({ status: 'error', message: 'No URL provided', url: null, source: 'nodejs_stealth_fetcher' }));
    process.exit(1);
}

if (!fs.existsSync(downloadsDir)){
    fs.mkdirSync(downloadsDir, { recursive: true });
}

(async () => {
    let browser;
    const screenshotPath = 'debug_screenshot.png';
    let downloadedPdfInfo = null;
    let interceptedPdfInfo = null; // For network interception

    try {
        browser = await chromium.launch({
            headless: false,
        });
        const context = await browser.newContext({
            userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            viewport: { width: 1920, height: 1080 },
            acceptDownloads: true
        });
        const page = await context.newPage();

        // Setup network interception
        await page.route('**/*', async (route, request) => {
            try {
                const requestUrl = request.url();
                const response = await route.fetch(); // Fetch the original response
                const headers = response.headers();
                const contentType = headers['content-type'];

                if (contentType && contentType.toLowerCase().includes('application/pdf')) {
                    const buffer = await response.buffer();
                    if (buffer && buffer.length > 0) {
                        const timestamp = new Date().toISOString().replace(/[.:T]/g, '-').slice(0, -5);
                        // Try to get a reasonable filename from the URL
                        let pdfFileName = requestUrl.substring(requestUrl.lastIndexOf('/') + 1).split('?')[0].split('#')[0];
                        if (!pdfFileName || !pdfFileName.toLowerCase().includes('.pdf')) {
                            pdfFileName = 'intercepted_document.pdf';
                        }
                        const safeOriginalNamePart = pdfFileName.replace(/[^a-zA-Z0-9_.-]/g, '_').substring(0, 50);
                        const filePath = path.join(downloadsDir, `${timestamp}-${safeOriginalNamePart}`);
                        
                        fs.writeFileSync(filePath, buffer);
                        interceptedPdfInfo = { 
                            path: filePath, 
                            resource_url: requestUrl, // URL of the PDF itself
                            final_url: page.url(), // URL of the page when PDF was intercepted
                            source: 'nodejs_stealth_fetcher_intercept'
                        };
                        // console.log(`Intercepted and saved PDF from ${requestUrl} to ${filePath}`);
                        // Decide whether to abort or continue. User wants to try continue first.
                        await route.continue(); 
                        return; // Important: exit handler after processing PDF
                    } else {
                        // console.error(`Intercepted PDF from ${requestUrl} but buffer was empty.`);
                        await route.continue();
                    }
                } else {
                    await route.continue();
                }
            } catch (error) {
                // console.error(`Error in route handler for ${request.url()}: ${error.message}. Continuing request.`);
                // If an error occurs in our handler, try to continue the request normally.
                // This is to prevent the request from hanging if our logic fails.
                try {
                    await route.continue();
                } catch (e) {
                    // console.error(`Failed to continue route after error: ${e.message}`);
                }
            }
        });

        page.on('download', async download => {
            try {
                const suggestedFilename = download.suggestedFilename();
                if (suggestedFilename.toLowerCase().endsWith('.pdf')) {
                    const timestamp = new Date().toISOString().replace(/[.:T]/g, '-').slice(0, -5);
                    const safeOriginalNamePart = suggestedFilename.replace(/[^a-zA-Z0-9_.-]/g, '_').substring(0, 50);
                    const filePath = path.join(downloadsDir, `${timestamp}-${safeOriginalNamePart}`);
                    await download.saveAs(filePath);
                    downloadedPdfInfo = { path: filePath, url: download.url() };
                } else {
                    await download.delete();
                }
            } catch (e) { /* console.error(`Error handling download: ${e.message}`); */ }
        });

        let response;
        try {
            // Initial navigation waiting for DOM content
            response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 }); // 60s for initial load
            // Wait for network to be idle after DOM is loaded
            await page.waitForLoadState('networkidle', { timeout: 90000 }); // 90s for network to settle
        } catch (navError) {
            // If primary navigation strategy fails, try a simpler one as a fallback
            // console.error(`Primary navigation failed: ${navError.message}. Trying fallback navigation.`);
            try {
                 response = await page.goto(url, { waitUntil: 'load', timeout: 120000 }); // 120s simple load
            } catch (fallbackNavError){
                // If both fail, rethrow the original error or a combined one
                throw fallbackNavError; // Or throw new Error(`Navigation failed: ${navError.message} & ${fallbackNavError.message}`);
            }
        }
        
        await page.waitForTimeout(Math.floor(Math.random() * 3000) + 2000); // 2-5s stabilization delay
        await page.screenshot({ path: screenshotPath, fullPage: true });

        const finalUrl = page.url(); // This is the URL after all redirects and navigation
        const responseHeaders = response ? response.headers() : {};
        const contentTypeHeader = responseHeaders['content-type'] || '';

        if (interceptedPdfInfo && interceptedPdfInfo.path) {
            console.log(JSON.stringify({
                status: 'pdf_downloaded',
                path: interceptedPdfInfo.path,
                final_url: interceptedPdfInfo.final_url, // Page URL when interception happened
                resource_url: interceptedPdfInfo.resource_url, // Actual PDF URL
                source: interceptedPdfInfo.source
            }));
        } else if (downloadedPdfInfo && downloadedPdfInfo.path) {
            console.log(JSON.stringify({
                status: 'pdf_downloaded',
                path: downloadedPdfInfo.path,
                final_url: finalUrl, // Page URL when download event fired
                resource_url: downloadedPdfInfo.url, // Actual PDF URL from download event
                source: 'nodejs_stealth_fetcher_event'
            }));
        } else if (finalUrl.toLowerCase().endsWith('.pdf') || contentTypeHeader.toLowerCase().includes('application/pdf')) {
            // Interception and download event failed, but page seems to be a PDF.
            // The previous page.context().request.get() attempt is known to cause 403s.
            // So, log this situation and fall back to HTML extraction.
            // console.warn(`Detected PDF page at ${finalUrl} but interception/download event failed. Falling back to HTML.`);
            const htmlContent = await page.content();
            let readableHtml = '';
            let mainContent = '';
            try {
                const doc = new JSDOM(htmlContent, { url: finalUrl });
                const reader = new Readability(doc.window.document);
                const article = reader.parse();
                if (article && article.content) {
                    readableHtml = article.content;
                    mainContent = article.textContent;
                }
            } catch (readabilityError) { /* ignore */ }
            console.log(JSON.stringify({
                status: 'success',
                html: readableHtml || htmlContent,
                main_text: mainContent,
                final_url: finalUrl,
                message: 'Detected PDF page, but network interception and download event failed. Returning page HTML.',
                source: 'nodejs_stealth_fetcher_inline_pdf_no_intercept_html'
            }));
        } else {
            const htmlContent = await page.content();
            let readableHtml = '';
            let mainContent = '';
            try {
                const doc = new JSDOM(htmlContent, { url: finalUrl });
                const reader = new Readability(doc.window.document);
                const article = reader.parse();
                if (article && article.content) {
                    readableHtml = article.content;
                    mainContent = article.textContent;
                }
            } catch (readabilityError) { /* ignore */ }
            
            console.log(JSON.stringify({
                status: 'success',
                html: readableHtml || htmlContent,
                main_text: mainContent,
                final_url: finalUrl,
                source: 'nodejs_stealth_fetcher_html'
            }));
        }

    } catch (error) {
        let errorMessage = error.message;
        if (error.stack) {
            errorMessage += '\\n' + error.stack;
        }
        console.error(JSON.stringify({
            status: 'error',
            message: errorMessage,
            url: url,
            source: 'nodejs_stealth_fetcher_main_error'
        }));
        process.exitCode = 1;
    } finally {
        if (browser) {
            await browser.close();
        }
    }
})();
