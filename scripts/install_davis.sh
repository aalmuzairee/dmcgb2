apt install -y unzip wget;
mkdir datasets;
cd datasets;
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip;
unzip DAVIS-2017-trainval-480p.zip;
rm DAVIS-2017-trainval-480p.zip;
