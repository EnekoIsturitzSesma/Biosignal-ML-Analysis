import requests
import os

def download_BCI_competition_2a_dataset():
    files = [
        "A01T.mat",
        "A01E.mat",
        "A02T.mat",
        "A02E.mat",
        "A03T.mat",
        "A03E.mat",
        "A04T.mat",
        "A04E.mat",
        "A05T.mat",
        "A05E.mat",
        "A06T.mat",
        "A06E.mat",
        "A07T.mat",
        "A07E.mat",
        "A08T.mat",
        "A08E.mat",
        "A09T.mat",
        "A09E.mat"
    ]

    for file in files:
        url = f"https://bnci-horizon-2020.eu/database/data-sets/001-2014/{file}"

        script_dir = os.getcwd()

        zip_path = os.path.join(
            script_dir,
            f"datasets/BCI_competition_2a/{file}"
        )

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"File downloaded to: {zip_path}")


download_BCI_competition_2a_dataset()