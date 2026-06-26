import os, urllib.request, json

BASE = "https://raw.githubusercontent.com/kennyvinente/optimization2026/main/materiais/AULA06"
API = "https://api.github.com/repos/kennyvinente/optimization2026/contents/materiais/AULA06"
ROOT = os.path.dirname(os.path.abspath(__file__))

def gh_api(path):
    req = urllib.request.Request(path, headers={"User-Agent": "claude"})
    return json.load(urllib.request.urlopen(req))

def dl(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "claude"})
    with urllib.request.urlopen(req) as r:
        open(dest, "wb").write(r.read())
    print("  ", os.path.relpath(dest, ROOT))

# Machine scheduling
print("Machine scheduling instances:")
for item in gh_api(f"{API}/machinescheduling_instances"):
    if item["type"] == "file":
        dl(item["download_url"], os.path.join(ROOT, "data", "machine", item["name"]))

# Job shop
print("Job shop:")
dl(f"{BASE}/jsplib_subset/instances.json", os.path.join(ROOT, "data", "jobshop", "instances.json"))
for item in gh_api(f"{API}/jsplib_subset/instances"):
    if item["type"] == "file":
        dl(item["download_url"], os.path.join(ROOT, "data", "jobshop", "instances", item["name"]))

# Flow shop
print("Flow shop:")
for item in gh_api(f"{API}/fssp_problems"):
    if item["type"] == "file":
        dl(item["download_url"], os.path.join(ROOT, "data", "flowshop", item["name"]))

print("DONE")
