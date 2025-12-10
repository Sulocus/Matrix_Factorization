#!/usr/bin/env python3
"""
Debug Streamlit UI - capture errors from page.
"""
import asyncio
from playwright.async_api import async_playwright

async def debug_ui():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Capture console errors
        errors = []
        page.on("console", lambda msg: errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda err: errors.append(f"[PAGE ERROR] {err}"))
        
        try:
            await page.goto('http://localhost:8501', wait_until='networkidle', timeout=15000)
            await page.wait_for_timeout(5000)
            
            # Check for Streamlit error
            error_elem = await page.query_selector('[data-testid="stException"]')
            if error_elem:
                error_text = await error_elem.inner_text()
                print("=== STREAMLIT EXCEPTION ===")
                print(error_text[:2000])
            
            # Check for any pre tags with errors
            pre_tags = await page.query_selector_all('pre')
            for pre in pre_tags:
                text = await pre.inner_text()
                if 'Error' in text or 'Traceback' in text:
                    print("=== ERROR IN PRE ===")
                    print(text[:2000])
            
            # Check page content
            body = await page.query_selector('body')
            body_html = await body.inner_html()
            print(f"\n=== PAGE CONTENT LENGTH: {len(body_html)} ===")
            
            if len(body_html) < 1000:
                print("Page seems empty, content:")
                print(body_html[:1000])
            
            # Console errors
            if errors:
                print("\n=== CONSOLE ERRORS ===")
                for e in errors[:10]:
                    print(e)
            
            await page.screenshot(path='/home/sucia/Sparse-Matrix/debug_screenshot.png', full_page=True)
            print("\nScreenshot saved to debug_screenshot.png")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(debug_ui())
