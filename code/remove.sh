#/bin/sh
echo "Removing Model Folder"
{ # try
    sudo chmod +777 $1
    sudo rm -r -f $1
    #save your output

} || { echo "Model path already not there"
} 