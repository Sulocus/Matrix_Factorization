#!/usr/bin/env python3
"""
Capture screenshot of Streamlit UI using Playwright.
"""
import asyncio
from playwright.async_api import async_playwright

async def capture_screenshot():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            # Try multiple ports
            for port in [8503, 8501, 8502]:
                try:
                    await page.goto(f'http://localhost:{port}', wait_until='networkidle', timeout=5000)
                    print(f"Connected to port {port}")
                    break
                except:
                    continue
            
            # Wait for content
            await page.wait_for_timeout(3000)
            
            # Check for error message
            error_text = await page.query_selector('pre.st-emotion-cache-')
            if error_text:
                content = await error_text.inner_text()
                print(f"ERROR FOUND:\n{content}")
            
            # Screenshot
            await page.screenshot(path='/home/sucia/Sparse-Matrix/ui_screenshot.png', full_page=True)
            print("Screenshot saved")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(capture_screenshot())
