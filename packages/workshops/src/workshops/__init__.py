import sys


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: workshops <lab>")
        print("Available labs: lab0, lab1, lab2, lab3, lab4, lab5, lab6")
        sys.exit(1)

    lab = args[0]
    if lab == "lab0":
        from workshops.lab0 import run_lab0

        run_lab0()
    elif lab == "lab1":
        from workshops.lab1 import run_lab1

        run_lab1()
    elif lab == "lab2":
        from workshops.lab2 import run_lab2

        run_lab2()
    elif lab == "lab3":
        from workshops.lab3 import run_lab3

        run_lab3()
    elif lab == "lab4":
        from workshops.lab4 import run_lab4

        run_lab4()
    elif lab == "lab5":
        from workshops.lab5 import run_lab5

        run_lab5()
    elif lab == "lab6":
        from workshops.lab6 import run_lab6

        run_lab6()
    else:
        print(f"Unknown lab: {lab}")
        print("Available labs: lab0, lab1, lab2, lab3, lab4, lab5, lab6")
        sys.exit(1)
