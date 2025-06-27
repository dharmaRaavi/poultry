import gdown

file_id = "1wpggf2pc2wtldElPWE-c0omBsfPdom29"
url = f"https://drive.google.com/uc?id={file_id}"
output = "model.h5"

gdown.download(url, output, quiet=False)
