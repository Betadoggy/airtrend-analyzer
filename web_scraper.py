import requests
from bs4 import BeautifulSoup
import urllib3

# 1. Disable SSL warnings for clean output
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = 'https://www.jll.com/en-us/insights/2026-aviation-trends'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

try:
    print(f"Connecting to...: {url}")
    
    # Use verify=False to bypass SSL certificate issues
    response = requests.get(url, headers=headers, verify=False, timeout=10)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract data
        page_title = soup.title.string if soup.title else "No Title Found"
        page_content = soup.get_text()
        
        # --- Save to Text File ---
        file_name = "crawling_result.txt"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"Source URL: {url}\n")
            f.write(f"Page Title: {page_title}\n")
            f.write("-" * 50 + "\n")
            f.write(page_content.strip())
        
        print(f"\n[Success] Data has been saved!")
        print(f"File Name: {file_name}")
        print(f"Page Title: {page_title}")
        print("-" * 30)
        
    else:
        print(f"\n[Failure] Status Code: {response.status_code}")

except Exception as e:
    print(f"\n[Error] An unexpected error occurred:")
    print(f"Details: {e}")