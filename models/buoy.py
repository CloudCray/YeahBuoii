import serial

class Buoy:
    def __init__(self, com_port, baud):
        self.port = serial.Serial(com_port, baud)
        self.values = [0, 0, 0, 0, 0, 0]

    def read_next(self):
        raw_row = self.port.readline()
        valid_row = validate(raw_row)
        if valid_row:
            if len(valid_row) == 6:
                self.values = valid_row
                print(self.values)
            else:
                print(valid_row)

def validate(raw_row):
    try:
        row_split = raw_row.decode("ascii").split(";")
        if len(row_split) == 1:
            return None
        else:
            row_split = raw_row.decode("ascii").split(";")[:-1]  # Extra ';' at the end
            if all([x.replace("-","").replace(".", "").isnumeric() for x in row_split]):
                return [float(x) for x in row_split]
            else:
                return None
    except:
        print(raw_row)
        return None