{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  packages = with pkgs; [
    (python3.withPackages (ps:
      with ps; [
        numpy
        matplotlib
      ]))
  ];
}
