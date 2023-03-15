# Create VirtuaEnv called zpoVenv
virtualenv -p python3 zpoVenv
# Run the VirtualEnv
source zpoVenv/bin/activate
# Install all required modules into the VirtualEnv
pip3 install -r requirements.txt
