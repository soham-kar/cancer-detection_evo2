
import requests
import json

def test_vep():
    server = "https://rest.ensembl.org"
    # Add nearest=1 to get nearest gene for intergenic
    ext = "/vep/homo_sapiens/region?nearest=symbol"
    headers = { "Content-Type" : "application/json", "Accept" : "application/json"}
    
    # Test variant from top of file: chr22 10510061 A T
    # 22 10510061 . A T 1
    
    # Test single GET
    # /vep/homo_sapiens/region/:region/:allele?nearest=symbol
    # Region: 22:10510061-10510061
    # Allele: T
    
    endpoint = server + "/vep/homo_sapiens/region/22:10510061-10510061:1/T"
    params = {"nearest": "symbol", "content-type": "application/json"}
    
    print(f"GET {endpoint}")
    r = requests.get(endpoint, headers=headers, params=params)
    
    if not r.ok:
        print(f"Error: {r.text}")
        return
        
    res = r.json()
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    test_vep()
