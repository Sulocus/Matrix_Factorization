#!/usr/bin/env python3
"""
Inspect Streamlit DOM structure to find correct CSS selectors for expander fix.
"""
import asyncio
from playwright.async_api import async_playwright

async def inspect_dom():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            await page.goto('http://localhost:8501', wait_until='networkidle')
            await page.wait_for_timeout(3000)
            
            # Find expander elements
            expanders = await page.query_selector_all('[data-testid="stExpander"]')
            print(f"Found {len(expanders)} expanders")
            
            if expanders:
                first = expanders[0]
                # Get inner HTML
                html = await first.inner_html()
                print("\n=== First Expander HTML ===")
                print(html[:1000])
                
                # Get computed styles of button
                button = await first.query_selector('div[role="button"]')
                if button:
                    styles = await button.evaluate('''el => {
                        const computed = window.getComputedStyle(el);
                        return {
                            display: computed.display,
                            padding: computed.padding,
                            position: computed.position,
                        };
                    }''')
                    print("\n=== Button Computed Styles ===")
                    print(styles)
                    
                    # Get button innerHTML
                    btn_html = await button.inner_html()
                    print("\n=== Button Inner HTML ===")
                    print(btn_html)
            
            # Screenshot with highlights
            await page.evaluate('''() => {
                document.querySelectorAll('[data-testid="stExpander"] > div[role="button"]').forEach(el => {
                    el.style.border = '2px solid red';
                });
            }''')
            await page.screenshot(path='/home/sucia/Sparse-Matrix/debug_expander.png', full_page=True)
            print("\nDebug screenshot saved to debug_expander.png")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(inspect_dom())
