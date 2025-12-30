import MyWrapper

def main() -> None:
    env = MyWrapper.MyEnv(verbose=False, render=True)
    #MyWrapper.MyEnv

    for _ in range(3):
        s, info = env.reset()

        for _ in range(500):
            s, r, term, trunc, info = env.step(0)

            if term or trunc: break

    print("finisched")



if __name__ == "__main__":
    main()