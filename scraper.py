import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from bs4.element import Comment


class ScrapeData:
    def __init__(self, url, website_name):
        self.url = url
        self.website_name = website_name

    # Filter visible text
    def tag_visible(self, element):
        if element.parent.name in ['style', 'script', 'head', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    # Extract all visible text from a rendered page
    async def get_text_from_js_page(self, url):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)
            await page.wait_for_load_state("networkidle")

            html = await page.content()
            await browser.close()

        soup = BeautifulSoup(html, 'html.parser')
        texts = soup.findAll(string=True)
        visible_texts = filter(self.tag_visible, texts)
        return "\n".join(t.strip() for t in visible_texts if t.strip())

    # Run
    async def run(self):
        visible_text = await self.get_text_from_js_page(self.url)

        with open(f"./scraped_policy/{self.website_name}.txt", "w", encoding="utf-8") as f:  # save file
            f.write(visible_text)


# To run the scraper in an async environment:
# async def main():
#     scrape_data = ScrapeData(url="https://www.apple.com/legal/privacy/en-ww/", website_name="Apple")
#     await scrape_data.run()

# if __name__ == "__main__":
#     asyncio.run(main())