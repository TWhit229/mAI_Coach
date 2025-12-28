import SwiftUI

struct BigRectButton: View {
    let title: String
    var systemImage: String? = nil

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(.black)
                .shadow(radius: 8, y: 4)
            HStack(spacing: 12) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.title2.weight(.semibold))
                        .foregroundStyle(.white)
                }
                Text(title)
                    .font(.title2.weight(.bold))
                    .foregroundStyle(.white)
            }
            .padding(.horizontal, 20)
        }
        .frame(height: 120)
        .contentShape(Rectangle()) // easier tapping
    }
}
