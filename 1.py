import torch

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA недоступна.")
        return

    device_count = torch.cuda.device_count()
    print(f"Количество GPU: {device_count}")

    for i in range(device_count):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")

    device = torch.device("cuda:0")
    x = torch.rand(1000, 1000, device=device)
    y = torch.rand(1000, 1000, device=device)
    z = torch.matmul(x, y)

    print("\nТестовое вычисление на GPU выполнено успешно.")
    print(f"Результат tensor device: {z.device}")
    print(f"Пример значения: {z[0, 0].item()}")

if __name__ == "__main__":
    main()