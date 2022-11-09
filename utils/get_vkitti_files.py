import os.path

def get_vkitti_files(dirName, exclude, ext, scenes, aug=None):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)

    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_vkitti_files(fullPath, exclude, ext, scenes, aug=aug)
        elif aug != None:
            scene = fullPath.split("/")[-6]
            if fullPath.split("/")[-5] not in exclude and fullPath.find("Camera_0") != -1 and fullPath.find(aug) != -1 and fullPath.find(ext) != -1 and scene in scenes:
                allFiles.append(fullPath)
        else:
            scene = fullPath.split("/")[-6]
            if fullPath.split("/")[-5] not in exclude and fullPath.find("Camera_0") != -1 and fullPath.find(ext) != -1 and scene in scenes:
                allFiles.append(fullPath)

    return allFiles