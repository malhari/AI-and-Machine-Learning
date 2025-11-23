# adventure_game.py
# A simple text-based adventure game where the player searches for a legendary treasure.

def start_game():
    """Starts the adventure game and handles the initial user choices."""
    print("\ Welcome to the Python Adventure Game!")
    print("Your quest is to find the legendary treasure hidden in the ancient land.\n")

    name = input("What is your name, brave explorer? ")
    print(f"\nWelcome, {name}! Your journey begins now...\n")

    while True:
        print("You stand before two paths:")
        print("1. Enter the dark forest")
        print("2. Explore the mysterious cave ")

        choice = input("Choose your path (1 or 2): ")

        if choice == "1":
            return forest_path()
        elif choice == "2":
            return cave_path()
        else:
            print("Invalid choice. Please choose again.\n")


def forest_path():
    """Forest scenario: Player chooses river or tree."""
    print("\n You enter the dark, dense forest.")
    print("The air is cold, and the sound of wildlife surrounds you.\n")
    print("You see two options:")
    print("1. Follow the river")
    print("2. Climb a tall tree to look around")

    choice = input("What do you want to do? (1 or 2): ")

    if choice == "1":
        print("\n You follow the river and discover a hidden waterfall.")
        print("Behind the waterfall, you find a glowing treasure chest!")
        print(" Congratulations! You found the legendary treasure!")
        return "win"

    elif choice == "2":
        print("\n You climb the tree, but a branch snaps!")
        print("You fall and injure yourself, making it impossible to continue.")
        print(" You lost the adventure.")
        return "lose"

    else:
        print("Invalid choice. You lose your way in the forest.")
        print(" You lost the adventure.")
        return "lose"


def cave_path():
    """Cave scenario: Player chooses torch or darkness."""
    print("\n You step into the mysterious cave.")
    print("It is cold, silent, and pitch black.\n")
    print("You have two options:")
    print("1. Light a torch")
    print("2. Proceed in the dark")

    choice = input("What do you want to do? (1 or 2): ")

    if choice == "1":
        print("\n You light a torch. The cave illuminates to reveal ancient markings.")
        print("As you proceed, you find an old chest resting on a stone altar.")
        print(" Congratulations! You found the legendary treasure!")
        return "win"

    elif choice == "2":
        print("\n You proceed in the dark...")
        print("You trip over a rock and fall into a deep pit!")
        print(" Your adventure ends here.")
        return "lose"

    else:
        print("Invalid choice. You wander aimlessly and get trapped.")
        print(" You lost the adventure.")
        return "lose"


def play_game():
    """Main game loop allowing replay."""
    while True:
        outcome = start_game()

        if outcome == "win":
            print("\n You completed the quest successfully!")
        else:
            print("\n Your quest has failed.")

        retry = input("\nDo you want to play again? (yes/no): ").lower()
        if retry not in ["yes", "y"]:
            print("\nThanks for playing the Python Adventure Game! Goodbye ")
            break


# Start the game
if __name__ == "__main__":
    play_game()
