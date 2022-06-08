def print_log(logger, msg):
    with open(logger, "a") as file:
        print(msg)
        file.write(msg+'\n')