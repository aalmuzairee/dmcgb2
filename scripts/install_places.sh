apt install -y wget;
mkdir datasets;
cd datasets;
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar;
tar -xvf places365standard_easyformat.tar; 
rm places365standard_easyformat.tar;