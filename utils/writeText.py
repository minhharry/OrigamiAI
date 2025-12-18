import os


def writeText(text, fileName, fileFolder, isAppend=False):
    if isAppend:
        filePath = os.path.join(fileFolder, fileName)
        if os.path.exists(filePath):
            with open(filePath, 'a') as f:
                f.write(text)
        else:
            with open(filePath, 'w') as f:
                f.write(text)
        return

    if not os.path.exists(fileFolder):
        os.makedirs(fileFolder)

    filePath = os.path.join(fileFolder, fileName)

    with open(filePath, 'w') as f:
        f.write(text)