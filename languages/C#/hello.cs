// Hello World program in C#.

// Code by Preston Hamlin

using System;

namespace csharp_hello
{
    class Greeting
    {
        protected string message;

        public Greeting(string m)
        {
            message = m;
        }

        public void Display()
        {
            Console.WriteLine(message);
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Greeting hello = new Greeting("Hello World!");
            hello.Display();
        }
    }
}
