/// Build script for compiling Protocol Buffers
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_file = "../CY_LLM_Backend/proto/ai_service.proto";
    let proto_dir = "../CY_LLM_Backend/proto";

    println!("cargo:rerun-if-changed={}", proto_file);

    // Compile protobuf files
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/generated")
        .compile(&[proto_file], &[proto_dir])?;

    Ok(())
}
