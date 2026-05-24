deadline urgency có thể encode trực tiếp thay vì để agent suy ra => Add remaining_times feature to HighLevelAgent
weight network & ll network chưa hội tụ ban đầu => Cho alpha và beta warm-up theo progress LowLevelAgent
thay vì failed rồi trả về fallback, hệ thống vẫn coi như thành công nên cộng reward dương => tách rõ fallback và pass
VGAE chỉ học topology reconstruction chứ không phải placement usefulness => Bổ sung auxiliary regression head
Bổ sung cơ chế chống collapse
Loại bỏ weight network
Tránh gradient competition