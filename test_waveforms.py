'''
import wfdb
import os
import requests
from bs4 import BeautifulSoup

# 1. Set base parameters
record_base = 'waves/p100/p10014354/81739927'
base_url = f'https://physionet.org/files/mimic4wdb/0.1.0/{record_base}/'
download_dir = f'mimic4wdb_download/{record_base}'

# 2. Scrape list of .hea files from PhysioNet page
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

hea_files = [
    a['href'].replace('.hea', '') for a in soup.find_all('a')
    if a['href'].endswith('.hea')
]

print(f"Found {len(hea_files)} record segments:")
for f in hea_files:
    print(f)

# 3. Download all segments
wfdb.dl_database(
    'mimic4wdb',
    'mimic4wdb_download',
    [f'{record_base}/{fname}' for fname in hea_files]
)

# 4. Load and plot the first record
if hea_files:
    first_record_path = os.path.join(download_dir, hea_files[0])
    record = wfdb.rdrecord(first_record_path)
    wfdb.plot_wfdb(record=record, title=f'Record: {hea_files[0]}')
else:
    print("No .hea files found.")
'''

import os
import requests
from bs4 import BeautifulSoup

def download_patient_data(patient_id, base_url='https://physionet.org/files/mimic4wdb/0.1.0/waves/p100/'):
    full_url = f"{base_url}{patient_id}/"
    print(f"🔍 Checking URL: {full_url}")

    # Fetch HTML from the directory page
    response = requests.get(full_url)
    if response.status_code != 200:
        print(f"❌ Failed to access: {full_url}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract folder names (e.g., 81739927, etc.)
    subfolders = [a['href'] for a in soup.find_all('a') if a['href'].strip('/').isdigit()]
    print(f"📁 Found subfolders: {subfolders}")

    for subfolder in subfolders:
        sub_url = f"{full_url}{subfolder}"
        sub_resp = requests.get(sub_url)
        if sub_resp.status_code != 200:
            print(f"❌ Failed to open: {sub_url}")
            continue

        sub_soup = BeautifulSoup(sub_resp.text, 'html.parser')
        file_links = [a['href'] for a in sub_soup.find_all('a') if not a['href'].endswith('/')]

        save_dir = os.path.join('downloaded', patient_id, subfolder.strip('/'))
        os.makedirs(save_dir, exist_ok=True)

        for file_name in file_links:
            file_url = f"{sub_url}{file_name}"
            save_path = os.path.join(save_dir, file_name)
            print(f"⬇️ Downloading: {file_url}")
            file_data = requests.get(file_url)
            with open(save_path, 'wb') as f:
                f.write(file_data.content)

    print(f"✅ Done downloading all files for {patient_id}")

# 🔽 Change this to your patient folder
download_patient_data('p10082591')
