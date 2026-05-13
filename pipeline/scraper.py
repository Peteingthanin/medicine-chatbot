import os
import requests
import urllib.parse
from bs4 import BeautifulSoup

def download_fda_pdfs(target_url, download_folder="fda_pdfs"):
    # 1. Create a folder for the downloaded PDFs
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
        print(f"Created folder: {download_folder}")

    print(f"Fetching data from: {target_url}")
    
    # 2. Get the HTML of the page
    response = requests.get(target_url)
    if response.status_code != 200:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return

    # 3. Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # 4. Find all PDF links
    # We look for <a> tags where the 'href' contains 'media.php' and '.pdf'
    pdf_links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if "media.php" in href and ".pdf" in href.lower():
            pdf_links.append(href)
            
    # Remove duplicate links just in case
    pdf_links = list(set(pdf_links))

    if not pdf_links:
        print("No PDF links found on this page.")
        return

    print(f"Found {len(pdf_links)} PDF(s) to download.\n")

    # 5. Download each file
    for index, pdf_url in enumerate(pdf_links, start=1):
        
        # Parse the URL to safely extract the 'name' parameter for our filename
        parsed_url = urllib.parse.urlparse(pdf_url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        if 'name' in query_params:
            # Get the actual filename (e.g., "46 Aciclovir_compressedtablet_PIL.pdf")
            filename = query_params['name'][0] 
        else:
            # Fallback just in case a name parameter is missing
            filename = f"document_{index}.pdf"
            
        # Clean the filename to prevent OS saving errors (remove invalid characters)
        valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        filename = "".join(c for c in filename if c in valid_chars)
        
        filepath = os.path.join(download_folder, filename)
        print(f"[{index}/{len(pdf_links)}] Downloading: {filename}")
        
        try:
            # Stream the file to disk
            pdf_response = requests.get(pdf_url, stream=True)
            pdf_response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print(" -> Success!")
            
        except requests.exceptions.RequestException as e:
            print(f" -> Failed to download. Error: {e}")

# Run the scraper
if __name__ == "__main__":
    url = "https://drug.fda.moph.go.th/drug-information"
    download_fda_pdfs(url)